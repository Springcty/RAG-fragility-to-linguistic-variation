import argparse
import os
import csv
import time
import logging
import asyncio
import pandas as pd
import aiolimiter
import openai
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============ Import your prompts.py ==============
# from prompts import PROMPTS  # Make sure prompts.py is in the same folder or adjust import path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

###############################################################################
# GPU & Model Setup
###############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Formality model
tokenizer_formality = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
model_formality = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker").to(device)

###############################################################################
# SBERT & Formality Helpers
###############################################################################
def compute_sbert_similarity(text1, text2):
    emb = sbert_model.encode([text1, text2], device=device)
    sim_val = util.cos_sim(emb[0], emb[1]).item()
    return float(sim_val)

def predict_formality_prob(text: str) -> float:
    """
    Returns the probability that `text` is 'formal'.
    We'll treat <0.5 as 'informal enough'.
    """
    inputs = tokenizer_formality(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model_formality(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    formal_probability = probs[0][1].item()
    return formal_probability

###############################################################################
# OPENAI ASYNC SETUP (from your provided script)
###############################################################################
import openai
import logging
import random
import numpy as np
import unicodedata
import re

# We assume you have an environment variable OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable must be set.")

# Construct an AsyncOpenAI client
client = openai.AsyncOpenAI(api_key=api_key)

async def openai_request(messages, model, temperature, max_tokens, top_p, limiter):
    """
    Calls openai, retrying up to 3 times on certain errors, using `limiter`.
    """
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
    """
    Single-call version to request a new rewriting from OpenAI.
    (For a single query at a time.)
    """
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
# Rewriting logic: "rewrite_query_until_valid"
###############################################################################
async def rewrite_query_until_valid(
    query_text: str,
    prompt_template: str,
    limiter,
    args,
    max_retries=5
):
    """
    Repeatedly calls openai with the prompt until we get formality<0.5 & sbert>0.7 
    or run out of retries. Returns (best_rewrite, attempts).
    If we never succeed, best_rewrite=None.
    """
    attempt = 0
    while attempt < max_retries:
        attempt += 1

        # Format the prompt from 'prompts.py' plus the original query
        full_prompt = prompt_template + "\n" + query_text
        messages = [{"role": "user", "content": full_prompt}]

        # Actually call openAI
        rewritten_text = await generate_openai_response(messages, args, limiter)
        if not rewritten_text:
            continue  # try again if we got an empty or None

        # Evaluate
        formality_val = predict_formality_prob(rewritten_text)
        sbert_val = compute_sbert_similarity(query_text, rewritten_text)

        if (formality_val < 0.5) and (sbert_val > 0.7):
            return rewritten_text, attempt
        # else keep trying

    return None, attempt  # we failed

###############################################################################
# Argument Parsing
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="""
    1) Reads top_5k.csv + a modified CSV (e.g. prompt_1_modified.csv).
    2) Computes SBERT for entire modified CSV => saves as prompt_1_modified_bert.csv.
    3) Partitions queries: pass vs missing/fail.
    4) Rewrites missing/fail queries using your openAI approach with prompts.py.
    5) Saves final 5k.
    6) Prints stats (missing, failing, rewriting attempts, etc.).
    """)
    parser.add_argument("--top5k_path", type=str, required=True,
                        help="Path to the top_5000.csv file (which has 5000 queries).")
    parser.add_argument("--modified_csv_path", type=str, required=True,
                        help="Path to the prompt_*_modified.csv file. We'll compute SBERT, rename to _bert, etc.")
    parser.add_argument("--output_final_csv", type=str, default="final_df.csv",
                        help="Where to save the final 5k output.")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max rewriting attempts per query.")
    parser.add_argument("--prompt_type", type=str, default="prompt_1",
                        help="Which prompt from PROMPTS to use.")
    parser.add_argument("--modification", type=str, default="formality",
                        choices=["formality","readability","politeness"],
                        help="Which modification to reference in PROMPTS.")
    # The usual OpenAI arguments from your script
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model name for rewriting.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for rewriting.")
    parser.add_argument("--max_response_tokens", type=int, default=100,
                        help="Max tokens for the rewriting.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="top_p for openai calls.")
    parser.add_argument("--requests_per_minute", type=int, default=150,
                        help="Rate limit for openAI.")
    return parser.parse_args()


###############################################################################
# Main
###############################################################################
async def main():
    args = parse_args()

    # =========== Load your PROMPTS from prompts.py ==============
    from prompts import PROMPTS  # Ensure prompts.py is in the same dir or adjust path

    if args.modification not in PROMPTS:
        raise ValueError(f"No prompts found for modification={args.modification}")
    if args.prompt_type not in PROMPTS[args.modification]:
        raise ValueError(f"No prompt_type={args.prompt_type} for modification={args.modification}")
    #log the prompt template 
    logging.info(f"Using prompt template: {PROMPTS[args.modification][args.prompt_type]}")

    # 1) Load top_5k
    if not os.path.isfile(args.top5k_path):
        logging.error(f"top_5k not found: {args.top5k_path}")
        return
    df_top5k = pd.read_csv(args.top5k_path)
    if "query" not in df_top5k.columns:
        logging.error("No 'query' column in top_5k CSV.")
        return

    top5k_queries = df_top5k["query"].unique().tolist()
    logging.info(f"Loaded top_5k with {len(top5k_queries)} unique queries (should be 5000).")

    # 2) Load the modified CSV
    if not os.path.isfile(args.modified_csv_path):
        logging.error(f"Modified CSV not found: {args.modified_csv_path}")
        return
    df_mod = pd.read_csv(args.modified_csv_path)

    if "query" not in df_mod.columns:
        logging.error("No 'query' column in the modified CSV.")
        return
    if "Response Formality Score" not in df_mod.columns:
        logging.error("No 'Response Formality Score' in the modified CSV.")
        return

    # 3) Compute SBERT across entire modified CSV if not present, then save as _bert
    if "sbert_similarity" not in df_mod.columns:
        logging.info("Computing SBERT similarity for each row in modified CSV...")
        sbert_vals = []
        for i in tqdm(range(len(df_mod)), desc="SBERT for Modified"):
            row = df_mod.iloc[i]
            q_text = str(row["query"])
            resp_text = str(row.get("response",""))
            sim = compute_sbert_similarity(q_text, resp_text)
            sbert_vals.append(sim)
        df_mod["sbert_similarity"] = sbert_vals

        base, ext = os.path.splitext(args.modified_csv_path)
        out_bert = base + "_bert" + ext
        df_mod.to_csv(out_bert, index=False)
        logging.info(f"Saved updated CSV with SBERT as: {out_bert}")

    # Convert df_mod to dict for quick lookup
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
            # found
            mrow = mod_dict[q]
            f_score = float(mrow["Response Formality Score"])
            s_score = float(mrow["sbert_similarity"])
            if (f_score < 0.5) and (s_score > 0.7):
                # pass => final
                new_row = dict(row)
                new_row["response"] = mrow.get("response","")
                new_row["Response Formality Score"] = f_score
                new_row["sbert_similarity"] = s_score
                final_rows.append(new_row)
            else:
                # failing
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

    # 5) Rewrite the work queries using your async approach + prompt from prompts.py
    # We'll build a rate limiter
    limiter = aiolimiter.AsyncLimiter(args.requests_per_minute)
    prompt_template = PROMPTS[args.modification][args.prompt_type]
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
        # attempts => how many times we called openAI
        total_rerun_count += (attempts - 1)

        if best_rewrite is None:
            # never passed => store a row with "failed"
            new_row = dict(wr)
            new_row["response"] = ""
            new_row["Response Formality Score"] = None
            new_row["sbert_similarity"] = None
            new_row["passed"] = False
            final_rows.append(new_row)
        else:
            # success => compute final formality + sbert
            form_prob = predict_formality_prob(best_rewrite)
            sim_val = compute_sbert_similarity(q, best_rewrite)

            new_row = dict(wr)
            new_row["response"] = best_rewrite
            new_row["Response Formality Score"] = form_prob
            new_row["sbert_similarity"] = sim_val
            new_row["passed"] = True
            final_rows.append(new_row)

    logging.info(f"Rewriting done. total re-run attempts beyond first: {total_rerun_count}")
    logging.info(f"final_rows length: {len(final_rows)} (should be 5000).")

    # 6) Save final
    df_final = pd.DataFrame(final_rows)
    output_jsonl_path = args.output_final_csv.replace(".csv", ".jsonl")
    df_final.to_json(output_jsonl_path, orient="records", lines=True)
    logging.info(f"Saved final 5k rows to {output_jsonl_path}")
    logging.info(f"Saved final 5k rows to {args.output_final_csv}")


if __name__ == "__main__":
    # We run the async main
    asyncio.run(main())

#Examples usage for PopQA:
# python query_rewriting/non_candidates.py   --top5k_path data/user_data/neelbhan/top_5k/PopQA/formality/top_5000.csv   --modified_csv_path data/user_data/tianyuca/QL_dataset/PopQA/formality/prompt_1_modified.csv   --output_final_csv data/user_data/tianyuca/QL_dataset/PopQA/formality/prompt_1_final_df.jsonl   --max_retries 5




#Examples usage for natural_questions:
