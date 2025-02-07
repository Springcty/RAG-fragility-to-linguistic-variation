import argparse
import os
import logging
import asyncio
import pandas as pd
import aiolimiter
import openai
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import re
import nltk
import random
import numpy as np
import unicodedata

# ============ Import your prompts.py ==============
from prompts import PROMPTS  # Adjust path if needed

# Make sure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

###############################################################################
# GPU & Model Setup
###############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# SBERT model (semantic similarity)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Formality model (RoBERTa)
tokenizer_formality = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
model_formality = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker").to(device)

# Politeness model (Intel/polite-guard)
politeness_pipeline = pipeline(
    'text-classification',
    model='Intel/polite-guard',
    device=0 if device == "cuda" else -1,
    return_all_scores=True
)

###############################################################################
# Metric Helpers
###############################################################################
def compute_sbert_similarity(text1, text2):
    emb = sbert_model.encode([text1, text2], device=device)
    sim_val = util.cos_sim(emb[0], emb[1]).item()
    return float(sim_val)

def predict_formality_prob(text: str) -> float:
    """
    Probability that text is formal (0..1). 
    We'll treat <0.5 as "informal enough" if modification=='formality'.
    """
    inputs = tokenizer_formality(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model_formality(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    return probs[0][1].item()  # Probability of "formal"

def predict_politeness_score(text: str) -> float:
    """
    Probability of text being polite + somewhat polite. 
    We'll treat >0.6 as polite enough if modification=='politeness'.
    """
    result = politeness_pipeline(text)[0]
    polite_score = 0.0
    for d in result:
        if "polite" in d['label'].lower():
            polite_score += d['score']
    return polite_score

def count_syllables(word: str) -> int:
    word = word.lower()
    syllable_count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith("e"):
        syllable_count -= 1
    return max(1, syllable_count)

def flesch_reading_ease(text: str) -> float:
    """
    Higher is more readable. 
    We'll treat >60 as passing if modification=='readability'.
    """
    sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
    words = re.findall(r'\w+', text)

    if not sentences or not words:
        return 0.0  # If no text, just return 0

    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = sum(count_syllables(w) for w in words) / len(words)

    return 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)

def compute_all_metrics(text: str) -> dict:
    """
    Return a dict with the four columns:
      - 'Response Formality Score'
      - 'Response Readability Score'
      - 'Response Politeness Score'
      - 'sbert_similarity' is computed elsewhere when we have the original query
        so we won't do it here. We'll do sbert separately.
    """
    return {
        "Response Formality Score": predict_formality_prob(text),
        "Response Readability Score": flesch_reading_ease(text),
        "Response Politeness Score": predict_politeness_score(text),
    }

def passes_threshold(modification: str, row_metrics: dict) -> bool:
    """
    Evaluate pass/fail based on the chosen modification's metric + sbert.
    row_metrics must contain:
      - 'Response Formality Score'
      - 'Response Readability Score'
      - 'Response Politeness Score'
      - 'sbert_similarity'
    """
    sbert_val = row_metrics["sbert_similarity"]
    if sbert_val <= 0.7:
        return False

    if modification == "formality":
        return row_metrics["Response Formality Score"] < 0.5
    elif modification == "readability":
        return row_metrics["Response Readability Score"] > 60.0
    elif modification == "politeness":
        return row_metrics["Response Politeness Score"] > 0.6
    else:
        raise ValueError(f"Unknown modification: {modification}")

###############################################################################
# OPENAI ASYNC SETUP
###############################################################################
import openai

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable must be set.")

client = openai.AsyncOpenAI(api_key=api_key)

async def openai_request(messages, model, temperature, max_tokens, top_p, limiter):
    async with limiter:
        for _ in range(3):  # 3 retries
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=1
                )
            except openai.APIConnectionError:
                logging.warning("API connection error, retrying in 10s...")
                await asyncio.sleep(10)
            except openai.RateLimitError:
                logging.warning("Rate limit exceeded, retrying in 30s...")
                await asyncio.sleep(30)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
            await asyncio.sleep(30)
        return None  # failed all retries

async def generate_openai_response(messages, args, limiter):
    resp = await openai_request(
        messages=messages,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_response_tokens,
        top_p=args.top_p,
        limiter=limiter
    )
    if resp is None:
        return ""
    return resp.choices[0].message.content.strip()

###############################################################################
# Rewriting logic
###############################################################################
async def rewrite_query_until_valid(
    query_text: str,
    prompt_template: str,
    limiter,
    args,
    max_retries=5
):
    """
    Repeatedly calls OpenAI with the prompt until it meets:
      - sbert_similarity > 0.7
      - relevant metric < threshold (see below)
      or runs out of retries.

    Returns:
        best_rewrite (str or None),
        attempts (int),
        final_metrics (dict or None) => contains 'Response Formality Score', 'Response Readability Score', 
                                        'Response Politeness Score', 'sbert_similarity'
    """
    attempt = 0
    while attempt < max_retries:
        attempt += 1

        # Build messages from prompt + original query
        full_prompt = prompt_template + "\n" + query_text
        messages = [{"role": "user", "content": full_prompt}]

        # Request a rewriting from OpenAI
        rewritten_text = await generate_openai_response(messages, args, limiter)
        if not rewritten_text:
            # If we got an empty or None response, try again
            continue

        # Compute semantic similarity with original query
        sbert_val = compute_sbert_similarity(query_text, rewritten_text)

        # If SBERT is not high enough, skip immediately
        if sbert_val <= 0.7:
            continue

        # Compute only the relevant metric
        if args.modification == "formality":
            # Threshold: < 0.5
            formality_val = predict_formality_prob(rewritten_text)
            if formality_val < 0.5:
                # Passed => now compute all metrics for final storage
                all_vals = compute_all_metrics(rewritten_text)
                all_vals["sbert_similarity"] = sbert_val
                return rewritten_text, attempt, all_vals

        elif args.modification == "readability":
            # Threshold: < 50
            readability_val = flesch_reading_ease(rewritten_text)
            if readability_val < 50.0:
                # Passed => compute all metrics
                all_vals = compute_all_metrics(rewritten_text)
                all_vals["sbert_similarity"] = sbert_val
                return rewritten_text, attempt, all_vals

        elif args.modification == "politeness":
            # Threshold: < 0.5
            polite_val = predict_politeness_score(rewritten_text)
            if polite_val < 0.5:
                # Passed => compute all metrics
                all_vals = compute_all_metrics(rewritten_text)
                all_vals["sbert_similarity"] = sbert_val
                return rewritten_text, attempt, all_vals

        # If none passed, go to next attempt

    # If we exhaust max_retries with no pass, return failure
    return None, attempt, None

###############################################################################
# Argument Parsing
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="""
    1) Reads top_5k.csv + a modified CSV (prompt_*_modified.csv).
    2) Ensures it has the columns:
        - 'Response Formality Score'
        - 'Response Readability Score'
        - 'Response Politeness Score'
        - 'sbert_similarity'
    3) Partitions pass/fail based on user-chosen modification + sbert>0.7.
    4) Rewrites fail queries; saves final with columns above + 'passed'.
    """)
    parser.add_argument("--top5k_path", type=str, required=True,
                        help="Path to the top_5000.csv with 'query' column.")
    parser.add_argument("--modified_csv_path", type=str, required=True,
                        help="Path to the prompt_*_modified.csv file. If columns are missing, we compute them.")
    parser.add_argument("--output_final_csv", type=str, default="final_df.csv",
                        help="Where to save the final 5k output CSV (also writes JSONL).")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max rewriting attempts per query.")
    parser.add_argument("--prompt_type", type=str, default="prompt_1",
                        help="Which prompt in PROMPTS to use.")
    parser.add_argument("--modification", type=str, default="formality",
                        choices=["formality","readability","politeness"],
                        help="Which metric to use for pass/fail checking.")
    # OpenAI arguments
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model name for rewriting.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for rewriting.")
    parser.add_argument("--max_response_tokens", type=int, default=100,
                        help="Max tokens for rewriting.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="top_p for openAI calls.")
    parser.add_argument("--requests_per_minute", type=int, default=150,
                        help="Rate limit for openAI requests.")
    return parser.parse_args()

###############################################################################
# Main
###############################################################################
async def main():
    args = parse_args()

    # Load prompt template
    if args.modification not in PROMPTS:
        raise ValueError(f"No prompts found for modification={args.modification}")
    if args.prompt_type not in PROMPTS[args.modification]:
        raise ValueError(f"No prompt_type={args.prompt_type} for modification={args.modification}")
    prompt_template = PROMPTS[args.modification][args.prompt_type]
    logging.info(f"Using prompt template for {args.modification}: {prompt_template}")

    # 1) Load top_5k
    if not os.path.isfile(args.top5k_path):
        logging.error(f"top_5k not found: {args.top5k_path}")
        return
    df_top5k = pd.read_csv(args.top5k_path)
    if "query" not in df_top5k.columns:
        logging.error("No 'query' column in top_5k CSV.")
        return

    top5k_queries = df_top5k["query"].unique().tolist()
    logging.info(f"Loaded top_5k with {len(top5k_queries)} unique queries (expected ~5000).")

    # 2) Load the modified CSV
    if not os.path.isfile(args.modified_csv_path):
        logging.error(f"Modified CSV not found: {args.modified_csv_path}")
        return
    df_mod = pd.read_csv(args.modified_csv_path)

    if "query" not in df_mod.columns:
        logging.error("No 'query' column in the modified CSV.")
        return

    # We'll ensure the 4 columns exist: 'Response Formality Score', 'Response Readability Score',
    # 'Response Politeness Score', 'sbert_similarity'.
    needed_cols = [
        "Response Formality Score",
        "Response Readability Score",
        "Response Politeness Score",
        "sbert_similarity"
    ]
    missing_cols = [col for col in needed_cols if col not in df_mod.columns]

    if missing_cols:
        logging.info(f"Missing columns {missing_cols}. Computing them now...")
        # We'll compute the metrics for each row
        new_data = {c: [] for c in missing_cols}

        for i in tqdm(range(len(df_mod)), desc="Computing Missing Metrics"):
            row = df_mod.iloc[i]
            q = str(row["query"])
            resp = str(row.get("response", ""))

            # Compute metrics for the response
            if "Response Formality Score" in missing_cols:
                new_data["Response Formality Score"].append(predict_formality_prob(resp))
            if "Response Readability Score" in missing_cols:
                new_data["Response Readability Score"].append(flesch_reading_ease(resp))
            if "Response Politeness Score" in missing_cols:
                new_data["Response Politeness Score"].append(predict_politeness_score(resp))
            if "sbert_similarity" in missing_cols:
                new_data["sbert_similarity"].append(compute_sbert_similarity(q, resp))

        # Append new columns
        for c in missing_cols:
            df_mod[c] = new_data[c]

        # Save an updated CSV for reference
        base, ext = os.path.splitext(args.modified_csv_path)
        out_path = base + "_bert" + ext
        df_mod.to_csv(out_path, index=False)
        logging.info(f"Saved updated CSV with newly computed metrics as: {out_path}")

    # Convert df_mod into a dict for quick lookup by query
    mod_dict = {}
    for i, row in df_mod.iterrows():
        mod_dict[row["query"]] = row

    # 4) Partition queries => final vs work
    final_rows = []
    work_rows = []
    missing_count = 0
    failing_count = 0

    for _, row in df_top5k.iterrows():
        q = row["query"]
        if q in mod_dict:
            mrow = mod_dict[q]
            row_metrics = {
                "Response Formality Score": float(mrow["Response Formality Score"]),
                "Response Readability Score": float(mrow["Response Readability Score"]),
                "Response Politeness Score": float(mrow["Response Politeness Score"]),
                "sbert_similarity": float(mrow["sbert_similarity"])
            }
            # Check if it passes
            if passes_threshold(args.modification, row_metrics):
                # Pass => final
                new_row = dict(row)
                new_row["response"] = mrow.get("response", "")
                new_row["Response Formality Score"] = row_metrics["Response Formality Score"]
                new_row["Response Readability Score"] = row_metrics["Response Readability Score"]
                new_row["Response Politeness Score"] = row_metrics["Response Politeness Score"]
                new_row["sbert_similarity"] = row_metrics["sbert_similarity"]
                new_row["passed"] = True
                final_rows.append(new_row)
            else:
                failing_count += 1
                work_rows.append(dict(row))
        else:
            # missing
            missing_count += 1
            work_rows.append(dict(row))

    passing_count = len(final_rows)
    logging.info(f"Passing queries: {passing_count}")
    logging.info(f"Missing queries: {missing_count}")
    logging.info(f"Failing queries: {failing_count}")
    logging.info(f"Total needing rewriting: {len(work_rows)}")

    # 5) Rewrite the work queries
    limiter = aiolimiter.AsyncLimiter(args.requests_per_minute)
    total_rerun_count = 0

    for wr in tqdm(work_rows, desc="Rewriting"):
        q = wr["query"]
        best_rewrite, attempts = await rewrite_query_until_valid(
            query_text=q,
            prompt_template=prompt_template,
            limiter=limiter,
            args=args,
            max_retries=args.max_retries
        )
        total_rerun_count += (attempts - 1)

        if best_rewrite is None:
            # never passed
            new_row = dict(wr)
            new_row["response"] = ""
            new_row["Response Formality Score"] = None
            new_row["Response Readability Score"] = None
            new_row["Response Politeness Score"] = None
            new_row["sbert_similarity"] = None
            new_row["passed"] = False
            final_rows.append(new_row)
        else:
            # success => compute the four columns
            metric_vals = compute_all_metrics(best_rewrite)
            sbert_val = compute_sbert_similarity(q, best_rewrite)
            new_row = dict(wr)
            new_row["response"] = best_rewrite
            new_row["Response Formality Score"] = metric_vals["Response Formality Score"]
            new_row["Response Readability Score"] = metric_vals["Response Readability Score"]
            new_row["Response Politeness Score"] = metric_vals["Response Politeness Score"]
            new_row["sbert_similarity"] = sbert_val
            new_row["passed"] = True
            final_rows.append(new_row)

    logging.info(f"Rewriting done. total re-run attempts (beyond first) = {total_rerun_count}")
    logging.info(f"final_rows length: {len(final_rows)} (should be 5000).")

    # 6) Save final
    df_final = pd.DataFrame(final_rows)
    df_final["passed"] = df_final["passed"].fillna(False)

    output_jsonl_path = args.output_final_csv.replace(".csv", ".jsonl")
    df_final.to_csv(args.output_final_csv, index=False)
    df_final.to_json(output_jsonl_path, orient="records", lines=True)

    logging.info(f"Saved final 5k rows to {args.output_final_csv}")
    logging.info(f"Saved final 5k rows to {output_jsonl_path}")

    logging.info("DONE")

if __name__ == "__main__":
    asyncio.run(main())