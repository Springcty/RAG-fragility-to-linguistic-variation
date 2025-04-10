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
import ast

from prompts import PROMPTS 


nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

sbert_model = SentenceTransformer('all-mpnet-base-v2', device=device)

tokenizer_formality = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
model_formality = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker").to(device)


politeness_pipeline = pipeline(
    'text-classification',
    model='Intel/polite-guard',
    device=0 if device == "cuda" else -1,
    return_all_scores=True
)


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
    return probs[0][1].item()  

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
        return 0.0  

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
        return row_metrics["Response Politeness Score"] > 0.5
    else:
        raise ValueError(f"Unknown modification: {modification}")


import openai

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable must be set.")

client = openai.AsyncOpenAI(api_key=api_key)

async def openai_request(messages, model, temperature, max_tokens, top_p, limiter):
    async with limiter:
        for _ in range(3):  
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
        return None  

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
      - (formality/politeness/readability) based on modification
    Returns: (best_rewrite [str or None], attempts [int])
    """
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        full_prompt = prompt_template + "\n" + query_text
        messages = [{"role": "user", "content": full_prompt}]
        rewritten_text = await generate_openai_response(messages, args, limiter)
        if not rewritten_text:
            continue

        sbert_val = compute_sbert_similarity(query_text, rewritten_text)
        if sbert_val <= 0.7:
            continue

        if args.modification == "politeness":
            valid = predict_politeness_score(rewritten_text) > 0.5
        elif args.modification == "readability":
            valid = flesch_reading_ease(rewritten_text) < 50.0
        elif args.modification == "formality":
            valid = predict_formality_prob(rewritten_text) < 0.5
        else:
            raise ValueError(f"Unknown modification: {args.modification}")
        if valid:
            return rewritten_text, attempt
    return None, attempt

def parse_args():
    parser = argparse.ArgumentParser(description="""
    This script reads a LARGE CSV (e.g. 8k or 10k queries),
    tries to find 5,000 queries that pass thresholds (with rewriting).
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
    parser.add_argument("--output_final_csv", type=str, default="None",
                        help="Where to save the 5k passing queries. Also writes JSONL.")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max rewriting attempts per query if it fails the initial check.")
    parser.add_argument("--modification", type=str, default="formality",
                        choices=["formality","readability","politeness"],
                        help="Which metric to use for pass/fail checking.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="If provided, only use the top N rows from the big CSV. Useful for testing.")
    parser.add_argument("--test_loading", action="store_true",
                        help="If set, only test loading the CSV (and applying sample_size) and exit.")
    # OpenAI arguments
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
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


async def main():
    args = parse_args()

    if args.test_loading:
        logging.info("Running in test loading mode.")
        test_loading(args)
        return

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
    logging.info("Goal: find 5k queries that pass thresholds with three prompt variants.")

    mod = args.modification
    if mod not in PROMPTS:
        raise ValueError(f"Missing prompts for modification: {mod}")
    for prompt in ["prompt_1", "prompt_2", "prompt_3"]:
        if prompt not in PROMPTS[mod]:
            raise ValueError(f"Missing {prompt} under PROMPTS['{mod}']")
    prompts_dict = {
        "prompt_1": PROMPTS[mod]["prompt_1"],
        "prompt_2": PROMPTS[mod]["prompt_2"],
        "prompt_3": PROMPTS[mod]["prompt_3"],
    }

    limiter = aiolimiter.AsyncLimiter(args.requests_per_minute)

    final_rows = []
    EXTRA_RETRIES = 5  
    prompt_success_counts = {"prompt_1": 0, "prompt_2": 0, "prompt_3": 0}
    combination_success_counts = {}

    for _, row in tqdm(df_big.iterrows(), desc="Scanning Big CSV", total=len(df_big)):
        if len(final_rows) >= 5000:
            break

        q = str(row["query"]).strip()
        if not q:
            continue

        valid_answer_found = False
        for col in ["answers", "answer", "possible_answers"]:
            if col in row:
                ans = row[col]
                if ans is None:
                    continue
                if isinstance(ans, list):
                    if not ans or any(item is None or not str(item).strip() for item in ans):
                        continue
                    else:
                        valid_answer_found = True
                        break
                elif str(ans).strip():
                    valid_answer_found = True
                    break
        if not valid_answer_found:
            continue

        results = {}
        for prompt_key in ["prompt_1", "prompt_2", "prompt_3"]:
            prompt_template = prompts_dict[prompt_key]
            best_rewrite, attempts = await rewrite_query_until_valid(
                query_text=q,
                prompt_template=prompt_template,
                limiter=limiter,
                args=args,
                max_retries=args.max_retries
            )
            results[prompt_key] = {
                "text": best_rewrite,  
                "attempts": attempts
            }

        valid_keys = [k for k, v in results.items() if v["text"] is not None]

        for key in valid_keys:
            prompt_success_counts[key] += 1
        combo_key = tuple(sorted(valid_keys))
        combination_success_counts[combo_key] = combination_success_counts.get(combo_key, 0) + 1

        if len(valid_keys) == 2:
            invalid_key = next(k for k in results if results[k]["text"] is None)
            prompt_template = prompts_dict[invalid_key]
            best_rewrite, extra_attempts = await rewrite_query_until_valid(
                query_text=q,
                prompt_template=prompt_template,
                limiter=limiter,
                args=args,
                max_retries=EXTRA_RETRIES
            )
            if best_rewrite is None:
                continue  
            else:
             
                results[invalid_key] = {
                    "text": best_rewrite,
                    "attempts": results[invalid_key]["attempts"] + extra_attempts
                }
                
                if invalid_key not in valid_keys:
                    prompt_success_counts[invalid_key] += 1

                valid_keys = [k for k, v in results.items() if v["text"] is not None]
                combo_key = tuple(sorted(valid_keys))
                combination_success_counts[combo_key] = combination_success_counts.get(combo_key, 0) + 1


        row_results = {}
        skip_this_query = False
        for prompt_key in ["prompt_1", "prompt_2", "prompt_3"]:
            best_rewrite = results[prompt_key]["text"]
            try:
                sbert_val = compute_sbert_similarity(q, best_rewrite)
                if args.modification == "politeness":
                    metric_val = predict_politeness_score(best_rewrite)
                    threshold = 0.5
                elif args.modification == "readability":
                    metric_val = flesch_reading_ease(best_rewrite)
                    threshold = 50.0
                elif args.modification == "formality":
                    metric_val = predict_formality_prob(best_rewrite)
                    threshold = 0.5
                else:
                    raise ValueError(f"Unknown modification: {args.modification}")
                
                # Safety check: if any rewrite fails thresholds, skip query
                if sbert_val <= 0.7 or metric_val >= threshold:
                    skip_this_query = True
                    break
                
                row_results[prompt_key] = {
                    "text": best_rewrite,
                    "sbert": sbert_val,
                    "metric_score": metric_val,
                    "attempts": results[prompt_key]["attempts"]
                }
            except:
                skip_this_query = True
        if skip_this_query:
            continue

        # All three prompts succeeded, so store the results
        new_row = dict(row)
        for prompt_key in ["prompt_1", "prompt_2", "prompt_3"]:
            new_row[f"{prompt_key}_text"] = row_results[prompt_key]["text"]
            new_row[f"{prompt_key}_sbert"] = row_results[prompt_key]["sbert"]
            new_row[f"{prompt_key}_metric_score"] = row_results[prompt_key]["metric_score"]
            new_row[f"{prompt_key}_attempts"] = row_results[prompt_key]["attempts"]
        final_rows.append(new_row)

    logging.info(f"Total queries processed (with any valid rewrite): {sum(combination_success_counts.values())}")
    logging.info(f"Successful rewrites per prompt: {prompt_success_counts}")
    logging.info("Successful combinations of prompts:")
    for combo, count in combination_success_counts.items():
        logging.info(f"  Prompts {combo}: {count}")

    # 7) Save final 5k (or fewer) queries
    df_final = pd.DataFrame(final_rows)

    if len(df_final) > 5000:
        df_final = df_final.head(5000)
    base_dir = os.path.dirname(args.output_final_csv)
    os.makedirs(base_dir, exist_ok=True) 
    output_jsonl_path = args.output_final_csv.replace(".csv", ".jsonl")
    df_final.to_json(output_jsonl_path, orient="records", lines=True)
    logging.info(f"Saved final passing queries to {output_jsonl_path}")
    
    id_col = None
    for col in ["id", "query_id"]:
        if col in df_final.columns:
            id_col = col
            break
    if id_col is None:
        df_final = df_final.reset_index().rename(columns={"index": "id"})
        id_col = "id"
    elif id_col == "query_id":
        df_final = df_final.rename(columns={"query_id": "id"})
        id_col = "id"


    answer_col = None
    possible_answer_cols = ["answers", "answer", "possible_answers"]
    answer_col = None

    for col in possible_answer_cols:
        if col in df_final.columns:
            answer_col = col
            break

    if answer_col is not None:
        if answer_col != "answers":
            df_final.rename(columns={answer_col: "answers"}, inplace=True)
            answer_col = "answers"
    else:
        df_final["answers"] = []
        answer_col = "answers"

    df_final["answers"] = df_final["answers"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') and x.strip().endswith(']')
        else x if isinstance(x, list) else [x]
    )

    passages_col = "passages" if "passages" in df_final.columns else None


    if args.modification == "politeness":
        df_final["query_linguistic_score"] = df_final.apply(lambda row: predict_politeness_score(row["query"]), axis=1)
    elif args.modification == "readability":
        df_final["query_linguistic_score"] = df_final.apply(lambda row: flesch_reading_ease(row["query"]), axis=1)
    elif args.modification == "formality":
        df_final["query_linguistic_score"] = df_final.apply(lambda row: predict_formality_prob(row["query"]), axis=1)
    else:
        raise ValueError(f"Unknown modification: {args.modification}")

    original_df = pd.DataFrame()
    original_df["id"] = df_final[id_col]
    original_df["question"] = df_final["query"]
    original_df["answers"] = df_final[answer_col] if answer_col in df_final.columns else ""
    original_df["passages"] = df_final[passages_col] if passages_col is not None else ""
    original_df["linguistics_score"] = df_final["query_linguistic_score"]


    modified_rows = []
    for _, row in df_final.iterrows():
        base_id = row["id"]
        for prompt_key in ["prompt_1", "prompt_2", "prompt_3"]:
            modified_query = row.get(f"{prompt_key}_text", "")
            ling_score = row.get(f"{prompt_key}_metric_score", None)
            sbert_sim = row.get(f"{prompt_key}_sbert", None)
            ans = row.get(answer_col, "")
            pas = row.get(passages_col, "") if passages_col is not None else ""
            modified_rows.append({
                "id": base_id,
                "question": modified_query,
                "answers": ans,
                "passages": pas,
                "linguistics_score": ling_score,
                "sbert_similarity": sbert_sim,
                "prompt": prompt_key
            })
    modified_df = pd.DataFrame(modified_rows)

    base_dir = os.path.dirname(args.output_final_csv)
    os.makedirs(base_dir, exist_ok=True)
    original_jsonl_path = os.path.join(base_dir, "original.jsonl")
    modified_jsonl_path = os.path.join(base_dir, "modified.jsonl")
    original_df.to_json(original_jsonl_path, orient="records", lines=True)
    modified_df.to_json(modified_jsonl_path, orient="records", lines=True)
    logging.info(f"Saved original queries JSONL to: {original_jsonl_path}")
    logging.info(f"Saved modified queries JSONL to: {modified_jsonl_path}")

if __name__ == "__main__":
    asyncio.run(main())

