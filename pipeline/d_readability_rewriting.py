import os
import json
import re
import argparse

import pandas as pd
import numpy as np

from openai import AsyncOpenAI
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

from utils.vllm_inference import vllm_inference

# Define an argument parser consisting of the following arguments: root_path, dataset, lingustics, and model


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, 
                    default='/data/group_data/maartens_lab_miis24/QL_dataset') 
parser.add_argument('--dataset', type=str, default='ms_marco') # ['popqa', 'entity_questions', 'ms_marco', 'natural_questions']
parser.add_argument('--split', type=str, default='validation') # ['validation', 'test']
parser.add_argument('--linguistics', type=str, default='readability') # ['readability', 'politeness']
parser.add_argument('--model', type=str, default='gpt-3.5-turbo') # ['gpt-3.5-turbo', 'gpt-4o-mini']
args = parser.parse_args()
print(args)
'''
python pipeline/d_readability_rewriting.py \
    --root_path /data/group_data/maartens_lab_miis24/QL_dataset \
    --dataset ms_marco --split validation \
    --linguistics readability \
    --model gpt-4o-mini
'''

if args.model == 'gpt-3.5-turbo':
    args.root_path = args.root_path.replace('QL_dataset', 'QL_dataset_gpt-3_5')
elif args.model == 'gpt-4o-mini':
    args.root_path = args.root_path.replace('QL_dataset', 'QL_dataset_gpt-4o')


# OpenAI Compatible Server
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Readability score
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
        return 0.0  # If no text, just return 0

    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = sum(count_syllables(w) for w in words) / len(words)

    return 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)

# SBERT Similarity Score
sbert_model = SentenceTransformer('all-mpnet-base-v2')
# sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
def compute_sbert_similarity(text1, text2):
    emb = sbert_model.encode([text1, text2])
    sim_val = util.cos_sim(emb[0], emb[1]).item()
    return float(sim_val)


# PopQA
def load_popqa():
    print('Loading PopQA dataset...')
    ds = load_dataset("akariasai/PopQA")
    df = ds['test'].to_pandas()
    df = df[['id', 'question', 'possible_answers']]
    df.rename(columns={'possible_answers': 'answers'}, inplace=True)
    return df

# EntityQuestion
def load_entityqa():
    print('Loading EntityQuestion dataset...')
    dev_dir = "/data/user_data/tianyuca/models/datasets/entityquestions/dev"

    # Initialize an empty list to store data
    all_data = []

    # Loop through all JSON files in the directory
    for filename in os.listdir(dev_dir):
        if filename.endswith(".json"):  # Only process JSON files
            filepath = os.path.join(dev_dir, filename)
            
            # Read JSON file
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)  # Load JSON as a list of dicts
                all_data.extend(data)  # Append to the main list

    df = pd.DataFrame(all_data)
    df['id'] = df.index
    return df

# MS MARCO
def load_ms_marco(split: str):
    ds = load_dataset("microsoft/ms_marco", "v1.1", split=split)
    df = pd.DataFrame(ds)
    df.rename(columns={'query': 'question', 'query_id': 'id'}, inplace=True)
    df = df[['id', 'question', 'answers', 'passages']]

    # filter rows with empty answers
    df = df[df['answers'].apply(lambda x: len(x) > 0)]
    
    # filter rows with questions shorter than 5 words
    df = df[df['question'].apply(lambda x: len(x.split()) >= 5)]
    
    return df

# Natural Questions
def load_natural_questions(split: str):
    data_dir = '/data/user_data/tianyuca/models/datasets/nq-open'
    if split == 'validation':
        file_path = os.path.join(data_dir, 'dev.json')
    elif split == 'test':
        file_path = os.path.join(data_dir, 'test.json')
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['data'])
    return df


# Load dataset
if args.dataset == 'popqa':
    df = load_popqa()
elif args.dataset == 'entity_questions':
    df = load_entityqa()
elif args.dataset == 'ms_marco':
    df = load_ms_marco(split=args.split)
elif args.dataset == 'natural_questions':
    df = load_natural_questions(split=args.split)
    

print('Filtering out questions with low readability score...')    
df['readability_score'] = df['question'].apply(lambda x: flesch_reading_ease(x))
df = df[df['readability_score'] > 60]
df = df.sort_values(by='readability_score', ascending=False)


df_v0 = pd.read_json('/data/group_data/maartens_lab_miis24/QL_dataset_gpt-4o/popqa/readability/readability_rewriting_validation_combined.jsonl', orient='records', lines=True)
# sample from df except for the ones that are already in df_v0
df = df[~df['id'].isin(df_v0['id'])]
df_sampled = df.sample(n=1500, random_state=42)

prompts = {
    'p1': '''1. Task Definition:
You are rewriting a query to make it significantly less readable while preserving the original semantic meaning as closely as possible.

2. Constraints & Goals:

- Flesch Reading Ease Score: The rewritten text must have a Flesch score below 60 (preferably below 50).
- Semantic Similarity: The rewritten text must have SBERT similarity > 0.7 compared with the original query.
- Length: The rewritten text must remain approximately the same length as the original query (±10%).
- Preserve Domain Terminology: Do not remove or drastically change domain-specific words, abbreviations, or technical terms (e.g., "IRS," "distance," etc.). 
- Abbreviation: Do not expand abbreviations unless the original query already used the expanded form.
- No New Information: You must not add additional details beyond what the original query states.
- Question Format: Retain the form of a question if the original is posed as a question.

3. How to Increase Complexity:

- Lexical Changes: Use advanced or academic synonyms only for common words. For domain or key terms (e.g., "distance," "IRS," "tax"), keep the original term or use a very close synonym if necessary to maintain meaning.
- Syntactic Complexity: Introduce passive voice, nominalizations, embedded clauses, and parenthetical or subordinate phrases. Ensure the sentence flow is more formal and convoluted without changing the core meaning.
- Redundancy & Formality: Employ circumlocution and excessively formal expressions (e.g., "due to the fact that" instead of "because") while avoiding any semantic drift.
- Dense, Indirect Construction: Favor longer phrases, indirect references, and wordiness. Avoid direct or simple phrasing.''',

    'p2': '''### **Task Description:**  
Transform a given query into a significantly less readable version while preserving its original semantic meaning as closely as possible.

### **Constraints & Goals:**  
- **Readability:** The rewritten text must have a **Flesch Reading Ease Score below 60**, preferably below 50.  
- **Semantic Similarity:** The rewritten text must achieve an **SBERT similarity score > 0.7** with the original query.  
- **Length Consistency:** The modified text should be **within ±10% of the original length**.  
- **Preserve Key Terminology:** **Do not alter domain-specific words, abbreviations, or technical jargon** (e.g., "IRS," "distance").  
- **Abbreviation Handling:** **Do not expand abbreviations** unless they are already expanded in the original query.  
- **Maintain Original Intent:** Do not add, remove, or alter the factual content of the query.  
- **Retain Question Structure:** If the input is a question, the output must also be a question.  

### **Techniques to Decrease Readability:**  
1. **Lexical Complexity:** Replace common words with **advanced, academic, or formal synonyms**, while keeping domain-specific terms unchanged.  
2. **Syntactic Complexity:** Introduce **passive voice, nominalizations, embedded clauses, or subordinate structures** to increase sentence density.  
3. **Redundancy & Formality:** Use **circumlocution, excessive formality, and indirect phrasing** (e.g., "in light of the fact that" instead of "because").  
4. **Dense Sentence Structure:** Prefer **wordy, indirect, and convoluted constructions** over direct phrasing.''',

    'p3': '''### **Objective:**  
You are tasked with **rewriting a given query to make it significantly less readable** while preserving its original semantic meaning with high fidelity.

### **Guiding Principles:**  
- **Readability Constraint:** The rewritten text must have a **Flesch Reading Ease Score of ≤60**, preferably ≤50.  
- **Semantic Integrity:** Ensure an **SBERT similarity score of at least 0.7** between the original and rewritten text.  
- **Length Tolerance:** Maintain an **approximate length deviation of no more than ±10%** from the original.  
- **Terminology Preservation:** Domain-specific terms (e.g., "IRS," "distance") **must remain intact** or be substituted only with **near-synonymous equivalents**.  
- **Abbreviation Handling:** If an abbreviation exists, **retain it as is** unless the original query explicitly expands it.  
- **Strict Content Preservation:** Do **not introduce any new information** or omit existing details.  
- **Question Retention:** If the input is a question, the reformulated output **must remain a question**.  

### **Techniques for Readability Reduction:**  
- **Lexical Sophistication:** Replace commonplace words with **more complex, formal, or technical alternatives** while maintaining clarity of meaning.  
- **Structural Density:** Employ **passive constructions, embedded clauses, and nominalized phrases** to increase syntactic complexity.  
- **Circumlocution & Wordiness:** Favor **verbose, indirect expressions** over concise phrasing (e.g., "with regard to" instead of "about").  
- **Elaborate Phrasing:** Use **multi-clause structures and intricate sentence formations** to reduce direct readability.''',
}

# build prompting questions
print('Building prompting questions...')
df_sampled['p1'] = df_sampled['question'].apply(lambda x: (prompts['p1'], f'Original Query: {x}\nLess Readable Query:'))
df_sampled['p2'] = df_sampled['question'].apply(lambda x: (prompts['p2'], f'Original Query: {x}\nLess Readable Query:'))
df_sampled['p3'] = df_sampled['question'].apply(lambda x: (prompts['p3'], f'Original Query: {x}\nLess Readable Query:'))

# run vllm_inference using the p1, p2, and p3 columns respectively and store the results in results_p1, results_p2, and results_p3
for p in ['p1', 'p2', 'p3']:
    print(f'Running vLLM inference with {p}...')
    results = vllm_inference(
        client=client,
        prompts=df_sampled[p].tolist(),
        model='gpt-3.5-turbo',
        temperature=1,
        max_tokens=100,
        top_p=0.9,
        requests_per_minute=150,
        num_responses_per_prompt=1,
    )
    
    df_sampled[f'{p}_response'] = results
    df_sampled[f'{p}_readability_score'] = df_sampled[f'{p}_response'].apply(lambda x: flesch_reading_ease(x))
    df_sampled[f'{p}_sbert_similarity'] = df_sampled.apply(lambda x: compute_sbert_similarity(x['question'], x[f'{p}_response']), axis=1)

    df_sampled_p_0 = df_sampled[df_sampled[f'{p}_readability_score'] < 60]
    df_sampled_p_1 = df_sampled[(df_sampled[f'{p}_sbert_similarity'] > 0.7)]
    df_sampled_p_2 = df_sampled[(df_sampled[f'{p}_readability_score'] < 60) & (df_sampled[f'{p}_sbert_similarity'] > 0.7)]
    print(f'Prompt {p}:', len(df_sampled_p_0), len(df_sampled_p_1), len(df_sampled_p_2))

# Re-generate using the prompt, if the other two passed but this one failed
print('Re-generating responses for query where two of the three prompts succeed ...')
max_regen_iterations = 5
prompt_keys = ['p1', 'p2', 'p3']
for i in range(max_regen_iterations):
    # Compute the success flags for each prompt
    for p in prompt_keys:
        df_sampled[f'{p}_success'] = (df_sampled[f'{p}_readability_score'] < 60) & (df_sampled[f'{p}_sbert_similarity'] > 0.7)
    regen_occurred = False
    
    # For each prompt, if the other two passed but this one failed, re-generate
    for p in prompt_keys:
        other_prompts = [q for q in prompt_keys if q != p]
        condition = (
            (~df_sampled[f'{p}_success'])
            & df_sampled[other_prompts[0] + '_success']
            & df_sampled[other_prompts[1] + '_success']
        )
        if condition.any():
            regen_occurred = True
            df_fail = df_sampled[condition]
            new_responses = vllm_inference(
                client=client,
                prompts=df_fail[p].tolist(),
                model='gpt-3.5-turbo',
                temperature=1,
                max_tokens=100,
                top_p=0.9,
                requests_per_minute=150,
                num_responses_per_prompt=1,
            )
            # Update the response and scores
            df_sampled.loc[df_fail.index, f'{p}_response'] = new_responses  
            df_sampled.loc[df_fail.index, f'{p}_readability_score'] = df_sampled.loc[df_fail.index, f'{p}_response'].apply(lambda x: flesch_reading_ease(x))
            df_sampled.loc[df_fail.index, f'{p}_sbert_similarity'] = df_sampled.loc[df_fail.index].apply(
                lambda x: compute_sbert_similarity(x['question'], x[f'{p}_response']), 
                axis=1
            )
    if not regen_occurred:
        break

# Compute the final success flags
for p in prompt_keys:
    df_sampled[f'{p}_success'] = (df_sampled[f'{p}_readability_score'] < 60) & (df_sampled[f'{p}_sbert_similarity'] > 0.7)
    df_sampled_p_0 = df_sampled[df_sampled[f'{p}_readability_score'] < 60]
    df_sampled_p_1 = df_sampled[(df_sampled[f'{p}_sbert_similarity'] > 0.7)]
    df_sampled_p_2 = df_sampled[(df_sampled[f'{p}_readability_score'] < 60) & (df_sampled[f'{p}_sbert_similarity'] > 0.7)]
    print(f'Prompt {p} after re-generation:', len(df_sampled_p_0), len(df_sampled_p_1), len(df_sampled_p_2))
    
# Filter out questions where all prompts succeeded
df_sampled_final = df_sampled[df_sampled['p1_success'] & df_sampled['p2_success'] & df_sampled['p3_success']]
print('Final Sample Size:', len(df_sampled_final))

# Save the final sampled dataset
print('Saving readability_rewriting.jsonl ...')
file_path = os.path.join(args.root_path, args.dataset, args.linguistics, f'readability_rewriting_{args.split}_random2.jsonl')
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df_sampled_final.to_json(file_path, orient='records', lines=True)
print('Done!')