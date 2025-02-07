import pandas as pd
import numpy as np
import nltk
from nltk.translate.gleu_score import sentence_gleu
from Levenshtein import distance as levenshtein_distance
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse

sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def compute_gleu(reference, hypothesis):
    ref_tokens = [nltk.word_tokenize(reference)]
    hyp_tokens = nltk.word_tokenize(hypothesis)
    return sentence_gleu(ref_tokens, hyp_tokens)

# Levenshtein distance (unchanged)
def compute_levenshtein_distance(ref, hyp):
    return levenshtein_distance(ref, hyp)

# BERTScore function
def compute_bert_score(ref, hyp, model_type="bert-base-uncased", lang="en"):
    """
    Returns the F1 BERTScore between the ref and hyp.
    - ref, hyp : single strings
    - model_type: specify which model to use for BERTScore
    - lang: language code (e.g., 'en' for English)
    """
    # bert_score expects list inputs: one reference, one hypothesis
    P, R, F1 = bert_score([hyp], [ref], model_type=model_type, lang=lang)
    return float(F1[0])  # get the scalar value

# SBERT similarity
# Load the SentenceTransformer model once (outside of function to avoid repeated loading)

def compute_sbert_similarity(text1, text2):
    """
    Computes semantic similarity using SBERT embeddings & cosine similarity.
    Returns a float between -1 and 1.
    """
    embeddings = sbert_model.encode([text1, text2])
    # embeddings[0] = embedding for text1, embeddings[1] = embedding for text2
    sim = util.cos_sim(embeddings[0], embeddings[1])
    return float(sim.item())

#Function to take 2 texts and see how much the modified query is longer than the original query per token
def compute_length_difference(original_query, modified_query):
    original_query_tokens = original_query.split()
    modified_query_tokens = modified_query.split()
    return len(modified_query_tokens) - len(original_query_tokens)

if __name__ == "__main__":

    # Load the SBERT model
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    #Create argument to parse input file
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to input file')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    transform_cols = ['edited_q']

    for col in transform_cols:
        # GLEU
        df[f'gleu_{col}'] = df.apply(lambda row: compute_gleu(row['query'], row[col]), axis=1)
        
        # Levenshtein
        df[f'levenshtein_{col}'] = df.apply(lambda row: compute_levenshtein_distance(row['query'], row[col]), axis=1)
        
        # BERTScore
        df[f'bert_score_{col}'] = df.apply(
            lambda row: compute_bert_score(row['query'], row[col], model_type="bert-base-uncased", lang="en"), 
            axis=1
        )
        
        # SBERT
        df[f'sbert_{col}'] = df.apply(lambda row: compute_sbert_similarity(row['query'], row[col]), axis=1)
        #length difference
        df[f'length_diff_{col}'] = df.apply(lambda row: compute_length_difference(row['query'], row[col]), axis=1)

    df.to_csv(args.output_file)
        