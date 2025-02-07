import re
import nltk
import os
import time
import logging
import argparse
import asyncio
import random
import unicodedata
import numpy as np
import pandas as pd
import aiolimiter
import openai
import json  # Needed for loading EntityQuestions
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import pipeline
from tqdm import tqdm

# Import prompt templates from a separate file
from prompts import PROMPTS

# (Assuming format_nq is defined elsewhere and imported if needed)
# from some_module import format_nq

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable must be set.")
client = openai.AsyncOpenAI(api_key=api_key)

def format_nq(nq_path):
    """
    Process the Natural Questions dataset and standardize columns to global requirements
    """
    dataset = load_dataset("google-research-datasets/natural_questions", "dev")
    nq_df = dataset['validation'].to_pandas()
    questions = [nq_df['question'][i]['text'] for i in range(len(nq_df))]
    nq_df['query'] = questions
    return nq_df
# Updated function to load dataset samples and return the full DataFrame.
def load_dataset_samples(dataset_name: str, sample_size: int, full_sampling: bool) -> pd.DataFrame:
    logging.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "natural_questions":
        # Assume format_nq returns a DataFrame.
        # If full_sampling is requested, we pass None (or an appropriate flag) to format_nq.
        df = format_nq(dataset_name)
            
    elif dataset_name == "ms_marco":
        dataset_ms = load_dataset('microsoft/ms_marco', 'v2.1', split='validation')
        df = dataset_ms.to_pandas()
        
    elif dataset_name == "PopQA":
        dataset = load_dataset("akariasai/PopQA")
        logging.info("Dataset loaded")
        train_dataset = dataset['test']
        df = train_dataset.to_pandas()
        # Rename 'question' column to 'query' for consistency.
        df.rename(columns={'question': 'query'}, inplace=True)
        
    elif dataset_name == "EntityQuestions":
        # Load the merged_dev.json file, extract questions and answers.
        with open("/home/neelbhan/QueryLinguistic/dataset/merged_dev.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = [item['question'] for item in data]
        answers = [item['answers'] for item in data]
        ids = list(range(len(questions)))
        df = pd.DataFrame(list(zip(ids, questions, answers)), columns=['id', 'query', 'answer'])
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # If sampling is requested (and sample_size is less than the total number of rows), sample the DataFrame.
    if not full_sampling and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return df


# Function to format OpenAI messages
def format_prompt(prompt: str) -> list:
    return [{"role": "user", "content": prompt}]


# Function to handle OpenAI API calls with retries
async def openai_request(messages, model, temperature, max_tokens, top_p, limiter):
    async with limiter:
        for _ in range(3):  # Retry mechanism
            try:
                return await client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, top_p=top_p, n=1
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


# Asynchronous batch OpenAI API request handler
async def generate_openai_responses(messages_list, args):
    limiter = aiolimiter.AsyncLimiter(args.requests_per_minute)
    tasks = [
        openai_request(
            msg, args.model, args.temperature, args.max_response_tokens, args.top_p, limiter
        )
        for msg in messages_list
    ]
    return await tqdm_asyncio.gather(*tasks)


def count_syllables(word: str) -> int:
    word = word.lower()
    syllable_count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith("e"):
        syllable_count -= 1
    return max(1, syllable_count)


def flesch_reading_ease(text: str) -> float:
    sentences = [s for s in re.split(r'[.!?]', text) if s]
    words = re.findall(r'\w+', text)

    if not sentences or not words:
        return None

    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = sum(count_syllables(w) for w in words) / len(words)

    return 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)


def calculate_formality(text: str) -> float:
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    pos_counts = {tag: 0 for tag in ["N", "J", "IN", "DT", "PRP", "VB", "RB", "UH"]}

    for _, tag in pos_tags:
        for key in pos_counts.keys():
            if tag.startswith(key):
                pos_counts[key] += 1

    total_words = len(words)
    if total_words == 0:
        return None

    formality_score = (
        sum([pos_counts["N"], pos_counts["J"], pos_counts["IN"], pos_counts["DT"]]) -
        sum([pos_counts["PRP"], pos_counts["VB"], pos_counts["RB"], pos_counts["UH"]]) + 100
    ) / 2
    return formality_score


def predict_formality(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    formal_probability = probabilities[0][1].item()

    return formal_probability


# Function to compute linguistic metrics and save results
# Load models once to avoid repeated initialization
def load_models():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
    logging.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    tokenizer_formality = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
    model_formality = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker").to(device)
    classifier_politeness = pipeline('text-classification', model='Intel/polite-guard', device=device)

    return tokenizer_formality, model_formality, classifier_politeness, device

# Batched inference for formality ranking using RoBERTa
def predict_formality_batch(tokenizer, model, texts, batch_size=16, device=-1):
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Formality Scoring"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        scores.extend(probabilities[:, 1].tolist())  # Extract probability for "formal" class
    return scores

# Batched inference for politeness scoring
def politeness_score_batch(classifier, texts, batch_size=16):
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Politeness Scoring"):
        batch = texts[i:i+batch_size]
        results = classifier(batch, return_all_scores=True)
        batch_scores = [res[0]['score'] + res[1]['score'] for res in results]  # Sum of polite + somewhat polite
        scores.extend(batch_scores)
    return scores

# Compute all metrics and save results, supporting full dataset sampling
def compute_and_save_results(df, output_path, full_sampling=False, sample_size=20):
    tokenizer_formality, model_formality, classifier_politeness, device = load_models()

    # Apply full sampling or limited sampling

    # Extract texts for scoring
    queries = df["query"].tolist()
    responses = df["response"].tolist()

    # Compute scores for queries
    df["Query Formality Score"] = predict_formality_batch(tokenizer_formality, model_formality, queries, device=device)
    df["Query Readability Score"] = df["query"].apply(flesch_reading_ease)
    df["Query Politeness Score"] = politeness_score_batch(classifier_politeness, queries)

    # Compute scores for responses
    df["Response Formality Score"] = predict_formality_batch(tokenizer_formality, model_formality, responses, device=device)
    df["Response Readability Score"] = df["response"].apply(flesch_reading_ease)
    df["Response Politeness Score"] = politeness_score_batch(classifier_politeness, responses)

    # Save results
    df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")


# Function to parse arguments
def get_args():
    parser = argparse.ArgumentParser(description="Query Modification and Evaluation")
    # Update dataset choices to include all supported datasets.
    parser.add_argument("--dataset", type=str, default="PopQA",
                        choices=["natural_questions", "ms_marco", "PopQA", "EntityQuestions"],
                        help="Dataset to use")
    parser.add_argument("--sample_size", type=int, default=20,
                        help="Number of queries to sample (ignored if --full_sampling is used)")
    parser.add_argument("--full_sampling", action="store_true",
                        help="Use full dataset instead of sampling")
    parser.add_argument("--modification", type=str, choices=["formality", "readability","politeness"], default="formality")
    parser.add_argument("--output_og", type=str,
                        help="Path for original query output CSV")
    parser.add_argument("--output_mod", type=str, required=True,
                        help="Path for modified query output CSV")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--max_response_tokens", type=int, default=100, help="Max response length")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p value")
    parser.add_argument("--requests_per_minute", type=int, default=150, help="API rate limit")
    # add argument for prompt type (e.g. prompt_1, prompt_2, etc.)
    parser.add_argument("--prompt_type", type=str, default="prompt_1", help="Prompt type to use")
    return parser.parse_args()


# Main Execution
if __name__ == "__main__":
    args = get_args()

    # Load the full dataset as a DataFrame.
    df_original = load_dataset_samples(args.dataset, args.sample_size, args.full_sampling)

    # Ensure the DataFrame has a 'query' column.
    if "query" not in df_original.columns:
        raise ValueError("The dataset does not contain a 'query' column.")

    # Extract the list of queries.
    queries = df_original["query"].tolist()

    # Retrieve a prompt template from the imported PROMPTS dictionary.
    if args.modification not in PROMPTS:
        raise ValueError(f"Unsupported modification type: {args.modification}")
    selected_prompt_template = PROMPTS[args.modification][args.prompt_type]
    logging.info(f"Selected prompt template: {selected_prompt_template}")
    

    # Construct the full prompts by appending the query text to the selected prompt template.
    prompts = [format_prompt(selected_prompt_template + "\n" + q) for q in queries]

    # Generate responses using the OpenAI API.
    responses = asyncio.run(generate_openai_responses(prompts, args))
    response_texts = [r.choices[0].message.content if r else "" for r in responses]

    # Add the responses as a new column in the original DataFrame.
    df_original["response"] = response_texts

    # # Optionally, you could save the original queries to a separate CSV if needed:
    # df_original.to_csv(args.output_og, index=False)
    # logging.info(f"Original queries saved to {args.output_og}")

    # Compute additional linguistic metrics and save the updated DataFrame (with responses and scores).
    df_original.to_csv(args.output_og, index=False)
    compute_and_save_results(df_original, args.output_mod)


## Example usage:
# python query_rewriting/test_rewrite_prompts.py --dataset PopQA --sample_size 20 --modification formality --output_og original_queries.csv --output_mod modified_queries.csv --prompt_type prompt_1