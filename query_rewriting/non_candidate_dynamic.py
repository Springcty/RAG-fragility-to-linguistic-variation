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
    We'll treat <0.5 as polite enough if modification=='politeness'.
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
    We'll treat <50 as passing if modification=='readability'.
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
    Return all three main metrics. We'll compute SBERT separately when we have the original query.
    """
    return {
        "Response Formality Score": predict_formality_prob(text),
        "Response Readability Score": flesch_reading_ease(text),
        "Response Politeness Score": predict_politeness_score(text),
    }

def test_loading(args):
    if not os.path.isfile(args.big_csv_path):
        logging.error(f"Big CSV not found: {args.big_csv_path}")
        return
    df_big = pd.read_csv(args.big_csv_path)
    if "query" not in df_big.columns:
        logging.error("No 'query' column in the big CSV.")
        return

    if args.sample_size is not None and args.sample_size > 0:
        df_big = df_big.head(args.sample_size)
        logging.info(f"Using only the top {args.sample_size} rows from the big CSV.")

    logging.info(f"Loaded big CSV with {len(df_big)} queries (sample_size applied if provided).")
    print(df_big.head())  # You can adjust what you want to print here.

###############################################################################
# THRESHOLD CHECK
###############################################################################
def passes_threshold(modification: str, row_metrics: dict) -> bool:
    """
    Evaluate pass/fail with new thresholds:
      - sbert > 0.7
      - formality < 0.5
      - readability < 50
      - politeness < 0.5
    """
    s_val = row_metrics["sbert_similarity"]
    if s_val <= 0.7:
        return False

    if modification == "formality":
        return row_metrics["Response Formality Score"] < 0.5
    elif modification == "readability":
        return row_metrics["Response Readability Score"] < 50.0
    elif modification == "politeness":
        return row_metrics["Response Politeness Score"] < 0.5
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
        for _ in range(3):  # Retry 3 times
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
        return None  # all retries failed

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
    For the chosen modification, we do up to max_retries attempts.
    Thresholds:
      - SBERT > 0.7
      - Formality < 0.5
      - Readability < 50
      - Politeness < 0.5

    Returns: (best_rewrite [str or None], attempts [int])
    """
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        full_prompt = prompt_template + "\n" + query_text
        messages = [{"role": "user", "content": full_prompt}]

        rewritten_text = await generate_openai_response(messages, args, limiter)
        if not rewritten_text:
            # If we got an empty response, try again
            continue

        sbert_val = compute_sbert_similarity(query_text, rewritten_text)
        if sbert_val <= 0.7:
            # fail => try again
            continue

        if args.modification == "formality":
            # pass if formality < 0.5
            if predict_formality_prob(rewritten_text) < 0.5:
                return rewritten_text, attempt
        elif args.modification == "readability":
            # pass if readability < 50
            if flesch_reading_ease(rewritten_text) < 50.0:
                return rewritten_text, attempt
        elif args.modification == "politeness":
            # pass if politeness < 0.5
            if predict_politeness_score(rewritten_text) < 0.5:
                return rewritten_text, attempt

        # otherwise, keep trying

    # if we run out of attempts
    return None, attempt


###############################################################################
# Argument Parsing
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="""
    This script reads a LARGE CSV (e.g. 8k or 10k queries),
    tries to find 5,000 queries that PASS thresholds (with rewriting).
    It stops as soon as it accumulates 5k pass.

    Metric thresholds:
        - SBERT > 0.7
        - Formality < 0.5
        - Readability < 50
        - Politeness < 0.5
    """)
    parser.add_argument("--big_csv_path", type=str, required=True,
                        help="Path to a big CSV with more than 5000 queries. Must have 'query' column.")
    parser.add_argument("--modified_csv_path", type=str, required=False, default=None,
                        help="(Optional) Path to the prompt_*_modified.csv with partial rewrites. If not given or not found, we'll skip it.")
    parser.add_argument("--output_final_csv", type=str, default="final_df.csv",
                        help="Where to save the 5k passing queries. Also writes JSONL.")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max rewriting attempts per query if it fails the initial check.")
    parser.add_argument("--prompt_type", type=str, default="prompt_1",
                        help="Which prompt in PROMPTS to use.")
    parser.add_argument("--modification", type=str, default="formality",
                        choices=["formality","readability","politeness"],
                        help="Which metric to use for pass/fail checking.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="If provided, only use the top N rows from the big CSV. Useful for testing.")
    parser.add_argument("--test_loading", action="store_true",
                        help="If set, only test loading the CSV (and applying sample_size) and exit.")
    # OpenAI arguments
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model for rewriting.")
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

    # 1) Load prompt template
    if args.test_loading:
        logging.info("Running in test loading mode.")
        test_loading(args)
        return
    if args.modification not in PROMPTS:
        raise ValueError(f"No prompts found for modification={args.modification}")
    if args.prompt_type not in PROMPTS[args.modification]:
        raise ValueError(f"No prompt_type={args.prompt_type} for modification={args.modification}")
    prompt_template = PROMPTS[args.modification][args.prompt_type]
    logging.info(f"Using prompt template for {args.modification}: {prompt_template}")

    # 2) Load big CSV (8k–10k queries) in order
    if not os.path.isfile(args.big_csv_path):
        logging.error(f"Big CSV not found: {args.big_csv_path}")
        return
    df_big = pd.read_csv(args.big_csv_path)
    if "query" not in df_big.columns:
        logging.error("No 'query' column in the big CSV.")
        return

    # NEW: Optionally slice df_big to the first sample_size rows.
    if args.sample_size is not None and args.sample_size > 0:
        df_big = df_big.head(args.sample_size)
        logging.info(f"Using only the top {args.sample_size} rows from the big CSV.")

    logging.info(f"Loaded big CSV with {len(df_big)} queries (sample_size applied if provided).")
    logging.info("Goal: find 5k queries that pass thresholds (with rewriting).")

    # 3) Try to load modified CSV if provided
    mod_dict = {}
    if args.modified_csv_path and os.path.isfile(args.modified_csv_path):
        df_mod = pd.read_csv(args.modified_csv_path)
        if "query" in df_mod.columns:
            for i, row in df_mod.iterrows():
                mod_dict[row["query"]] = row
        else:
            logging.warning("No 'query' column in the modified CSV; ignoring file.")
    else:
        if args.modified_csv_path:
            logging.warning(f"Modified CSV not found at: {args.modified_csv_path}. Skipping it.")
        else:
            logging.info("No modified_csv_path provided. Proceeding without it.")

    # We'll store the final passing queries
    final_rows = []

    # Async limiter for OpenAI
    limiter = aiolimiter.AsyncLimiter(args.requests_per_minute)

    # 4) Single pass: iterate top->bottom until we have 5k passing
    rewriting_attempts = 0
    total_used = 0
    for _, row in tqdm(df_big.iterrows(), desc="Scanning Big CSV", total=len(df_big)):
        if len(final_rows) >= 5000:
            break  # we have enough passing queries

        q = str(row["query"])
        # 4a) If in mod_dict, check if it passes now
        mod_passed = False
        resp_text = ""
        fm = None
        rd = None
        pl = None
        sb = None

        if q in mod_dict:
            # We have some partial rewriting stored
            # Evaluate it with new thresholds
            drow = mod_dict[q]
            resp_text = str(drow.get("response", ""))

            # We compute sbert for safety if missing
            sb = compute_sbert_similarity(q, resp_text)

            # Based on your new thresholds, formality <0.5, readability <50, politeness <0.5
            fm = predict_formality_prob(resp_text)
            rd = flesch_reading_ease(resp_text)
            pl = predict_politeness_score(resp_text)

            row_metrics = {
                "Response Formality Score": fm,
                "Response Readability Score": rd,
                "Response Politeness Score": pl,
                "sbert_similarity": sb
            }
            if passes_threshold(args.modification, row_metrics):
                # This is a pass => no rewriting needed
                mod_passed = True

        # 4b) If not mod_passed => attempt rewriting
        if not mod_passed:
            best_rewrite, attempts = await rewrite_query_until_valid(
                query_text=q,
                prompt_template=prompt_template,
                limiter=limiter,
                args=args,
                max_retries=args.max_retries
            )
            rewriting_attempts += (attempts - 1)

            if best_rewrite is None:
                # skip this query, move on
                continue
            else:
                # success => fill in final metrics
                resp_text = best_rewrite
                sb = compute_sbert_similarity(q, resp_text)
                fm = predict_formality_prob(resp_text)
                rd = flesch_reading_ease(resp_text)
                pl = predict_politeness_score(resp_text)

        # 4c) We either had it pass from the mod file or from rewriting
        new_row = dict(row)
        new_row["response"] = resp_text
        new_row["Response Formality Score"] = fm
        new_row["Response Readability Score"] = rd
        new_row["Response Politeness Score"] = pl
        new_row["sbert_similarity"] = sb
        new_row["passed"] = True
        final_rows.append(new_row)
        total_used += 1

    # 5) Finished scanning. Check if we got 5k
    logging.info(f"Total passing queries found: {len(final_rows)}")
    if len(final_rows) < 5000:
        logging.warning("We could NOT find 5k passing queries. We'll save what we have.")
    else:
        logging.info("Successfully found 5k passing queries. Stopping now.")

    # 6) Save final 5k
    df_final = pd.DataFrame(final_rows)
    # If we have more than 5k, we take top 5k
    if len(df_final) > 5000:
        df_final = df_final.head(5000)

    output_jsonl_path = args.output_final_csv.replace(".csv", ".jsonl")
    df_final.to_csv(args.output_final_csv, index=False)
    df_final.to_json(output_jsonl_path, orient="records", lines=True)

    logging.info(f"Saved final passing queries to {args.output_final_csv}")
    logging.info(f"Saved final passing queries to {output_jsonl_path}")
    logging.info(f"Total rewriting attempts (beyond first) = {rewriting_attempts}")
    logging.info("DONE")

if __name__ == "__main__":
    asyncio.run(main())