import re
import nltk
from nltk import pos_tag, word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import argparse
import asyncio
import logging
import os
from datasets import load_dataset
import random
import aiolimiter
import openai
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import unicodedata

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')



def count_syllables(word):
    """Counts syllables in a word based on the number of vowel groups."""
    word = word.lower()
    syllable_count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith("e"):
        syllable_count -= 1
    return max(1, syllable_count)

def flesch_reading_ease(text):
    """Calculates the Flesch Reading Ease Score (FRES) for the given text."""
    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if s]
    words = re.findall(r'\w+', text)

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(word) for word in words)

    if num_sentences == 0 or num_words == 0:
        return None  

    avg_words_per_sentence = num_words / num_sentences
    avg_syllables_per_word = num_syllables / num_words

    fres = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    return fres

def predict_formality(tokenizer,model,text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    formal_probability = probabilities[0][1].item()

    return formal_probability

if __name__ == "__main__":

    # Load the SBERT model
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str, help='Path to input file')
    args.add_argument('--output_file', type=str, help='Path to output file')
    args = args.parse_args()
    df = pd.read_csv(args.input_file)

    model_name = "s-nlp/roberta-base-formality-ranker"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    formal_bert=[]
    readability= []
    for i in range(len(df)):
        text= df['query'][i]
        formal_bert.append(predict_formality(tokenizer=tokenizer,model=model,text=text))
        readability.append(flesch_reading_ease(text))

    df['formality_score'] = formal_bert
    df['readability_score'] = readability

    df.to_csv(args.output_file)


    ## Command to run the script and explain it
    # python form_read_len_calc.py --input_file /home/neelbhan/QueryLinguistic/query_rewriting/analysing_metrics_datasets/sampled_df.csv --output_file /home/neelbhan/QueryLinguistic/query_rewriting/analysing_metrics_datasets/sampled_df_frl.csv
    # This script calculates the formality and readability score for each question in the dataset and saves the output in a CSV file.
