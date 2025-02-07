import os
import csv
import argparse
import logging
import re
import nltk
import json
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

###############################################################################
# GPU check
###############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if device == "cuda" else -1
logging.info(f"Using device: {device}")

###############################################################################
# NLTK downloads
###############################################################################
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

###############################################################################
# Load each dataset fully (no sampling), focusing on queries only
###############################################################################
def format_nq():
    """
    Process the Natural Questions dataset, returning a DataFrame with 'query' column.
    """
    dataset = load_dataset("google-research-datasets/natural_questions", "dev")
    nq_df = dataset['validation'].to_pandas()
    questions = [nq_df['question'][i]['text'] for i in range(len(nq_df))]
    nq_df['query'] = questions
    return nq_df

def load_dataset_full(dataset_name: str) -> pd.DataFrame:
    logging.info(f"Loading dataset: {dataset_name} (full)")
    if dataset_name == "natural_questions":
        df = format_nq()
    elif dataset_name == "ms_marco":
        dataset_ms = load_dataset('microsoft/ms_marco', 'v2.1', split='validation')
        df = dataset_ms.to_pandas()
    elif dataset_name == "PopQA":
        dataset = load_dataset("akariasai/PopQA")
        logging.info("PopQA loaded")
        test_dataset = dataset['test']
        df = test_dataset.to_pandas()
        df.rename(columns={'question': 'query'}, inplace=True)
    elif dataset_name == "EntityQuestions":
        with open("/home/neelbhan/QueryLinguistic/dataset/merged_dev.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = [item['question'] for item in data]
        answers = [item['answers'] for item in data]
        ids = list(range(len(questions)))
        df = pd.DataFrame(list(zip(ids, questions, answers)), columns=['id', 'query', 'answer'])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if "query" not in df.columns:
        raise ValueError("The dataset does not contain a 'query' column.")

    # Reset index
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Loaded {len(df)} total rows.")
    return df

###############################################################################
# Flesch Reading Ease
###############################################################################
def count_syllables(word: str) -> int:
    word = word.lower()
    syllable_count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith("e"):
        syllable_count -= 1
    return max(1, syllable_count)

def flesch_reading_ease(text: str) -> float:
    sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
    words = re.findall(r'\w+', text)
    if not sentences or not words:
        return 0.0
    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = sum(count_syllables(w) for w in words) / len(words)
    return 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)

###############################################################################
# Formality and Politeness models
###############################################################################
def load_models():
    tokenizer_formality = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
    model_formality = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker").to(device)
    classifier_politeness = pipeline(
        "text-classification",
        model='Intel/polite-guard',
        device=device_id
    )
    return tokenizer_formality, model_formality, classifier_politeness

def predict_formality_batch(tokenizer, model, texts, batch_size=16):
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Formality Scoring"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        # Probability for "formal" class is index 1
        scores.extend(probs[:, 1].tolist())
    return scores

def politeness_score_batch(classifier, texts, batch_size=16):
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Politeness Scoring"):
        batch = texts[i:i+batch_size]
        results = classifier(batch, return_all_scores=True)
        batch_scores = []
        for res in results:
            # Example of res: [{'label': 'polite', 'score': 0.9}, {'label': 'impolite', 'score': 0.1}]
            polite_score = 0.0
            for d in res:
                if "polite" in d['label'].lower():
                    polite_score += d['score']
            batch_scores.append(polite_score)
        scores.extend(batch_scores)
    return scores

###############################################################################
# Compute missing query metrics
###############################################################################
def compute_query_metrics_if_missing(df: pd.DataFrame):
    """
    Compute 'Query Formality Score', 'Query Readability Score', 'Query Politeness Score'
    if they are not already in the DataFrame.
    """
    needed_cols = {
        "Query Formality Score": "formality",
        "Query Readability Score": "readability",
        "Query Politeness Score": "politeness"
    }
    to_compute = [col for col in needed_cols if col not in df.columns]
    if not to_compute:
        logging.info("All query-level scores already present. Skipping computation.")
        return df

    tokenizer_formality, model_formality, classifier_politeness = load_models()

    queries = df["query"].fillna("").tolist()

    if "Query Formality Score" in to_compute:
        logging.info("Computing Query Formality...")
        q_formality = predict_formality_batch(tokenizer_formality, model_formality, queries)
        df["Query Formality Score"] = q_formality

    if "Query Readability Score" in to_compute:
        logging.info("Computing Query Readability...")
        df["Query Readability Score"] = [flesch_reading_ease(q) for q in tqdm(queries, desc="Query Readability")]

    if "Query Politeness Score" in to_compute:
        logging.info("Computing Query Politeness...")
        q_polite = politeness_score_batch(classifier_politeness, queries)
        df["Query Politeness Score"] = q_polite

    return df

###############################################################################
# Main script
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Select top 5000 queries from a dataset for a chosen modification, queries only.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["natural_questions", "ms_marco", "PopQA", "EntityQuestions"],
                        help="Which dataset to load.")
    parser.add_argument("--modification", type=str, required=True,
                        choices=["formality", "readability", "politeness"],
                        help="Which metric to sort by on the query side.")
    args = parser.parse_args()

    # 1) Load data
    df = load_dataset_full(args.dataset)

    # 2) Compute missing query-level metrics
    df = compute_query_metrics_if_missing(df)

    # 3) Sorting logic:
    #   - formality => Query Formality Score (descending)
    #   - readability => Query Readability Score (descending)
    #   - politeness => Query Politeness Score (ascending)
    if args.modification == "formality":
        sort_col = "Query Formality Score"
        ascending = False
    elif args.modification == "readability":
        sort_col = "Query Readability Score"
        ascending = False
    else:  # politeness => least polite => ascending
        sort_col = "Query Politeness Score"
        ascending = True

    df.sort_values(by=sort_col, ascending=ascending, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4) Take top 5000
    top_5k = df.head(5000)
    logging.info(f"Selected top {len(top_5k)} queries for modification={args.modification} (sorted by {sort_col}).")

    # 5) Save to data/user_data/neelbhan/top_5k/{DATASET}/{MODIFICATION}/top_5000.csv
    out_dir = os.path.join("data", "user_data", "neelbhan", "top_5k", args.dataset, args.modification)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "top_5000.csv")
    top_5k.to_csv(out_path, index=False)
    logging.info(f"Top 5000 saved to: {out_path}")

if __name__ == "__main__":
    main()

# Example usage all commands for NQ:
# python query_rewriting/filter_5000.py --dataset natural_questions --modification formality
# python query_rewriting/filter_5000.py --dataset natural_questions --modification readability
# python query_rewriting/filter_5000.py --dataset natural_questions --modification politeness
# Example usage all commands for MS Marco:
# python query_rewriting/filter_5000.py --dataset ms_marco --modification formality
# python query_rewriting/filter_5000.py --dataset ms_marco --modification readability
# python query_rewriting/filter_5000.py --dataset ms_marco --modification politeness
# Example usage all commands for EntityQuestions:
# python query_rewriting/filter_5000.py --dataset EntityQuestions --modification formality
# python query_rewriting/filter_5000.py --dataset EntityQuestions --modification readability
# python query_rewriting/filter_5000.py --dataset EntityQuestions --modification politeness


