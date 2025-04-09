#!/usr/bin/env python
import random
import string
import argparse
import pandas as pd
import json
import logging
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from easynmt import EasyNMT
from datasets import load_dataset
from grammar_metric_calc import (  # Importing functions from grammar_metrics.py
    compute_gleu,
    compute_levenshtein_distance,
    compute_bert_score,
    compute_sbert_similarity,
    compute_length_difference
)

from nltk.translate.gleu_score import sentence_gleu
from Levenshtein import distance as levenshtein_distance
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

# Ensure necessary NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)


def format_nq(nq_path):
    """
    Process the Natural Questions dataset and standardize columns to global requirements
    """
    dataset = load_dataset("google-research-datasets/natural_questions", "dev")
    nq_df = dataset['validation'].to_pandas()
    questions = [nq_df['question'][i]['text'] for i in range(len(nq_df))]
    nq_df['query'] = questions
    return nq_df
    
def load_dataset_samples(dataset_name: str, sample_size: int = 1000, full_sampling: bool = False) -> pd.DataFrame:
    logging.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "natural_questions":
        df = format_nq(dataset_name)  # Assuming format_nq() returns a DataFrame
            
    elif dataset_name == "ms_marco":
        dataset_ms = load_dataset('microsoft/ms_marco', 'v2.1', split='validation')
        df = dataset_ms.to_pandas()
        
    elif dataset_name == "PopQA":
        dataset = load_dataset("akariasai/PopQA")
        logging.info("Dataset loaded")
        train_dataset = dataset['test']
        df = train_dataset.to_pandas()
        df.rename(columns={'question': 'query'}, inplace=True)  # Rename column for consistency

    elif dataset_name == "EntityQuestions":
        with open("/home/neelbhan/QueryLinguistic/dataset/merged_dev.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = [item['question'] for item in data]
        answers = [item['answers'] for item in data]
        ids = list(range(len(questions)))
        df = pd.DataFrame(list(zip(ids, questions, answers)), columns=['id', 'query', 'answer'])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not full_sampling and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logging.info(f"Sampled {sample_size} items from {dataset_name}")

    return df


def edit_word(word, edit_probability=0.2, num_edits=1):
    if random.random() < edit_probability:
        operation = random.choice(['addition', 'substitution', 'removal'])
        for _ in range(num_edits):
            if operation == 'addition':
                pos = random.randint(0, len(word))
                extra_char = random.choice(string.ascii_lowercase)
                word = word[:pos] + extra_char + word[pos:]
            elif operation == 'substitution' and len(word) > 1:
                pos = random.randint(0, len(word)-1)
                new_char = random.choice(string.ascii_lowercase)
                word = word[:pos] + new_char + word[pos+1:]
            elif operation == 'removal' and len(word) > 1:
                pos = random.randint(0, len(word)-1)
                word = word[:pos] + word[pos+1:]
    return word

def edit_sentence(sentence, edit_probability=0.2, num_edits=1):
    tokens = word_tokenize(sentence)
    new_tokens = [edit_word(token, edit_probability, num_edits) if token.isalpha() else token for token in tokens]
    return ' '.join(new_tokens)


def get_synonyms(word, pos):
    synonyms = set()
    for syn in wordnet.synsets(word):
        if syn.pos() == pos:
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return synonyms

def pos_edit(sentence, probability=0.2, error_types=("verb", "preposition", "noun")):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    new_tokens = tokens[:]
    
    for i, (word, tag) in enumerate(tagged):
        if "verb" in error_types and tag.startswith('VB') and random.random() < probability:
            new_tokens[i] = word[:-3] if word.endswith('ing') else word + 'ing'
        elif "preposition" in error_types and tag.startswith('IN') and random.random() < probability:
            prepositions = ['in', 'on', 'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through']
            new_tokens[i] = random.choice(prepositions)
        elif "noun" in error_types and tag.startswith('NN') and random.random() < probability:
            synonyms = get_synonyms(word, 'n')
            if synonyms:
                new_tokens[i] = random.choice(list(synonyms))
    
    return ' '.join(new_tokens)


def back_translate(text_list, source_lang="en", pivot_lang="af", model_name="opus-mt"):
    model = EasyNMT(model_name)
    translated = model.translate(text_list, source_lang=source_lang, target_lang=pivot_lang)
    back_translated = model.translate(translated, source_lang=pivot_lang, target_lang=source_lang)
    return back_translated


def main():
    parser = argparse.ArgumentParser(
        description="Apply text editing (character, POS, back translation) and compute metrics comparing edited text to original."
    )

    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to load (e.g., 'natural_questions').")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--sample_size', type=int, default=1000, help="Number of samples to take from the dataset (default: 1000).")
    parser.add_argument('--full_sampling', action='store_true', help="Process the full dataset instead of sampling.")

    parser.add_argument('--apply_char', action='store_true', help="Apply character-level editing.")
    parser.add_argument('--apply_pos', action='store_true', help="Apply POS-based editing.")
    parser.add_argument('--apply_back', action='store_true', help="Apply back translation.")

    parser.add_argument('--char_prob', type=float, default=0.2, help="Probability for character-level editing (default: 0.2).")
    parser.add_argument('--num_edits', type=int, default=1, help="Number of character edits per word (default: 1).")
    parser.add_argument('--pos_prob', type=float, default=0.2, help="Probability for POS-based editing (default: 0.2).")
    parser.add_argument('--pivot_lang', type=str, default="af", help="Pivot language for back translation.")
    parser.add_argument('--translate_model_name', type=str, default="opus-mt", help="Translation model name (default: 'opus-mt').")

    args = parser.parse_args()

    df = load_dataset_samples(args.dataset_name, sample_size=args.sample_size, full_sampling=args.full_sampling)
    if "query" not in df.columns:
        raise ValueError("The dataset does not contain a 'query' column.")

    original_texts = df["query"].astype(str)
    edited_cols = []

    if args.apply_char:
        df['edited_query_char'] = original_texts.apply(lambda text: edit_sentence(text, edit_probability=args.char_prob, num_edits=args.num_edits))
        edited_cols.append('edited_query_char')

    if args.apply_pos:
        df['edited_query_pos'] = original_texts.apply(lambda text: pos_edit(text, probability=args.pos_prob))
        edited_cols.append('edited_query_pos')

    if args.apply_back:
        back_translated = back_translate(original_texts.tolist(), source_lang="en", pivot_lang=args.pivot_lang, model_name=args.translate_model_name)
        df['back_translated_query'] = back_translated
        edited_cols.append('back_translated_query')

    for col in edited_cols:
        df[f'gleu_{col}'] = df.apply(lambda row: compute_gleu(row['query'], row[col]), axis=1)
        df[f'levenshtein_{col}'] = df.apply(lambda row: compute_levenshtein_distance(row['query'], row[col]), axis=1)
        df[f'bert_score_{col}'] = df.apply(lambda row: compute_bert_score(row['query'], row[col]), axis=1)
        df[f'sbert_{col}'] = df.apply(lambda row: compute_sbert_similarity(row['query'], row[col]), axis=1)
        df[f'length_diff_{col}'] = df.apply(lambda row: compute_length_difference(row['query'], row[col]), axis=1)

    df.to_csv(args.output_file, index=False)
    print(f"Processed file saved to {args.output_file}")

if __name__ == "__main__":
    main()