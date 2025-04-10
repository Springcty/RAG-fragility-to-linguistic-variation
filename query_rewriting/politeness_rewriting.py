import os
import json
import argparse
import ast

import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from datasets import load_dataset, Dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

from utils.vllm_inference import vllm_inference

# Define an argument parser consisting of the following arguments: root_path, dataset, lingustics, and model


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, 
                    default='/data/QL_dataset') 
parser.add_argument('--dataset', type=str, default='ms_marco') # ['popqa', 'entity_questions', 'ms_marco', 'natural_questions']
parser.add_argument('--split', type=str, default='validation') # ['validation', 'test']
parser.add_argument('--linguistics', type=str, default='politeness') # ['readability', 'politeness']
parser.add_argument('--model', type=str, default='gpt-4o-mini') # ['gpt-3.5-turbo', 'gpt-4o-mini']
args = parser.parse_args()

args.root_path = os.path.join(args.root_path, args.model)
print(args)
'''
python pipeline/d_politeness_rewriting.py \
    --root_path /data/QL_dataset \
    --dataset ms_marco --split validation \
    --linguistics politeness \
    --model gpt-4o-mini
'''


# OpenAI Compatible Server
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Load the politeness classifier
classifier = pipeline(
    'text-classification',
    model='Intel/polite-guard',
    device=0,
)

def politeness_score(query: str):
    # return the sum of the scores for 'polite' and 'somewhat polite'
    result = classifier(query, top_k=None)
    # result: [{'label': 'somewhat polite', 'score': 0.9407630562782288}, {'label': 'neutral', 'score': 0.04077668860554695}, {'label': 'polite', 'score': 0.017881780862808228}, {'label': 'impolite', 'score': 0.0005785255925729871}]
    label = result[0]['label']
    score = 0
    for r in result:
        if r['label'] in ['polite', 'somewhat polite']:
            score += r['score']    
    return (score, label)

def politeness_score_batch(batch, prompt=None):
    if prompt:
        queries = batch[f'{prompt}_response']
    else:
        queries = batch['question']
    results = classifier(queries, top_k=None)
    
    scores = []
    labels = []
    for result in results:
        score = sum(r['score'] for r in result if r['label'] in ['polite', 'somewhat polite'])
        scores.append(score)
        labels.append(result[0]['label'])
    
    if prompt:
        batch[f'{prompt}_politeness_score'] = scores
        batch[f'{prompt}_politeness_label'] = labels
    else:
        batch['politeness_score'] = scores
        batch['politeness_label'] = labels
    return batch


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
    # format the answers
    df['answers'] = df['answers'].apply(lambda x: ast.literal_eval(x))
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
    # df = df[df['question'].apply(lambda x: len(x.split()) >= 5)]
    
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


# HotpotQA
def load_hotpotqa():
    print('Loading HotpotQA dataset...')
    data_path = '/data/MultiHop/hotpot_qa/hotpot_dev_distractor_v1.json'
    df_full = pd.read_json(data_path)
    df = df_full[['_id', 'question', 'answer']]
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
elif args.dataset == 'hotpotqa':
    df = load_hotpotqa()


# filter out questions with politeness_score < 0.5 and sort by politeness_score (ascending)
print('Filtering out questions with politeness_score < 0.5')
df_datasets = Dataset.from_pandas(df, preserve_index=False)
df_datasets = df_datasets.map(politeness_score_batch, batched=True, batch_size=128)
df = df_datasets.to_pandas()
df = df[df['politeness_score'] < 0.5]
df = df.sort_values(by='politeness_score', ascending=True)

# Sample 7500 questions
df_sampled = df.iloc[:7500].copy()

# Load prompts
prompts = {
    'p1': '''### Task: Rewrite Queries to Sound More Polite and Courteous  

Rephrase the given query into a more polite, respectful, and considerate version while preserving its original intent. The output should reflect a natural, well-mannered tone suitable for professional or friendly interactions. The generated query should be a single sentence.

#### Critical Rules:  
- Use a variety of politeness techniques, including warm greetings, indirect requests, and expressions of gratitude.  
- Avoid robotic or overly formal constructions—make it sound naturally courteous, warm and friendly.  
- Do not always start your sentence with 'Could you please tell'. Use emotional undertones and specific attempts at politeness.
- Maintain the original meaning without unnecessary embellishment.
- Do not start the generated query with 'I hope you are ...' or end with a single 'Thank you' sentence. Generate only a single polite query sentence.
''',

    'p2': '''### Task: Enhance the Courtesy of a Given Query
    
Transform the provided query into a more respectful, friendly, and warm version, ensuring it conveys respect and warmth while keeping the original intent intact. The reworded request should sound engaging, professional, and well-mannered. The generated query should be a single sentence.

### Key Considerations:
- Use a mix of politeness techniques, including indirect phrasing, friendly introductions, and appreciative language.
- Keep the tone natural—avoid overly rigid or formal wording that feels robotic.
- Vary sentence structures instead of defaulting to "Could you please...". Use emotional undertones and specific attempts at politeness.
- Maintain the original meaning while subtly enhancing the request's politeness and friendliness.
- Avoid beginning the generated query with 'I hope you are...' or concluding it with a separate 'Thank you.' sentence. Generate only one polite query sentence.
''',

    'p3': '''### Task: Refining Queries for Politeness and Warmth

Transform a given query into a more courteous, engaging, and warm request while ensuring it retains the original intent. The revised version should sound friendly, professional, and respectful. The generated query should be a single sentence.

### Guidelines:
- Incorporate politeness techniques such as indirect requests, warm introductions, and appreciative language.
- Ensure the tone is natural—avoid excessive formality that feels robotic.
- Diversify sentence structures rather than defaulting to "Could you please...". Use emotional undertones and specific attempts at politeness.
- Subtly enhance warmth and professionalism while preserving clarity and intent.
- Avoid beginning the generated query with 'I hope you are ...' or concluding it with a standalone 'Thank you' sentence. Generate only one polite query sentence.
''',
}


# build prompting questions
print('Building prompting questions...')
df_sampled['p1'] = df_sampled['question'].apply(lambda x: (prompts['p1'], f'Original Query: {x}\nPolite Query:'))
df_sampled['p2'] = df_sampled['question'].apply(lambda x: (prompts['p2'], f'Original Query: {x}\nPolite Query:'))
df_sampled['p3'] = df_sampled['question'].apply(lambda x: (prompts['p3'], f'Original Query: {x}\nPolite Query:'))

# run vllm_inference using the p1, p2, and p3 columns respectively and store the results in results_p1, results_p2, and results_p3
for p in ['p1', 'p2', 'p3']:
    print(f'Running vLLM inference with {p}...')
    results = vllm_inference(
        client=client,
        prompts=df_sampled[p].tolist(),
        model=args.model,
        temperature=1,
        max_tokens=100,
        top_p=0.9,
        requests_per_minute=150,
        num_responses_per_prompt=1,
    )
    
    df_sampled[f'{p}_response'] = results
    df_sampled_dataset = Dataset.from_pandas(df_sampled, preserve_index=False)
    df_sampled_dataset = df_sampled_dataset.map(
        politeness_score_batch, 
        batched=True, 
        batch_size=128,
        fn_kwargs={'prompt': p},
    )
    df_sampled = df_sampled_dataset.to_pandas()
    df_sampled[f'{p}_sbert_similarity'] = df_sampled.apply(lambda x: compute_sbert_similarity(x['question'], x[f'{p}_response']), axis=1)
    
    df_sampled_p_0 = df_sampled[df_sampled[f'{p}_politeness_score'] > 0.5]
    df_sampled_p_1 = df_sampled[df_sampled[f'{p}_sbert_similarity'] > 0.7]
    df_sampled_p_2 = df_sampled[(df_sampled[f'{p}_politeness_score'] > 0.5) & (df_sampled[f'{p}_sbert_similarity'] > 0.7)]
    print(f'Prompt {p}:', len(df_sampled_p_0), len(df_sampled_p_1), len(df_sampled_p_2))

# Re-generate using the prompt, if the other two passed but this one failed
print('Re-generating responses for query where two of the three prompts succeed ...')
max_regen_iterations = 5
prompt_keys = ['p1', 'p2', 'p3']
for i in range(max_regen_iterations):
    # Compute the success flags for each prompt
    for p in prompt_keys:
        df_sampled[f'{p}_success'] = (df_sampled[f'{p}_politeness_score'] > 0.5) & (df_sampled[f'{p}_sbert_similarity'] > 0.7)
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
                model=args.model,
                temperature=1,
                max_tokens=100,
                top_p=0.9,
                requests_per_minute=150,
                num_responses_per_prompt=1,
            )
            # Update the response and scores
            df_sampled.loc[df_fail.index, f'{p}_response'] = new_responses  
            df_sampled.loc[df_fail.index, [f'{p}_politeness_score', f'{p}_politeness_label']] = df_sampled.loc[df_fail.index, f'{p}_response'].apply(
                lambda x: pd.Series(politeness_score(x))
            )
            df_sampled.loc[df_fail.index, f'{p}_sbert_similarity'] = df_sampled.loc[df_fail.index].apply(
                lambda x: compute_sbert_similarity(x['question'], x[f'{p}_response']), 
                axis=1
            )
    if not regen_occurred:
        break

# Compute the final success flags
for p in prompt_keys:
    df_sampled[f'{p}_success'] = (df_sampled[f'{p}_politeness_score'] > 0.5) & (df_sampled[f'{p}_sbert_similarity'] > 0.7)
    df_sampled_p_0 = df_sampled[df_sampled[f'{p}_politeness_score'] > 0.5]
    df_sampled_p_1 = df_sampled[(df_sampled[f'{p}_sbert_similarity'] > 0.7)]
    df_sampled_p_2 = df_sampled[(df_sampled[f'{p}_politeness_score'] > 0.5) & (df_sampled[f'{p}_sbert_similarity'] > 0.7)]
    print(f'Prompt {p} after re-generation:', len(df_sampled_p_0), len(df_sampled_p_1), len(df_sampled_p_2))
    
# Filter out questions where all prompts succeeded
df_sampled_final = df_sampled[df_sampled['p1_success'] & df_sampled['p2_success'] & df_sampled['p3_success']]
print('Final Sample Size:', len(df_sampled_final))

# Save the final sampled dataset
print('Saving politeness_rewriting.jsonl ...')
if args.dataset == 'hotpotqa':
    file_path = '/data/MultiHop/hotpot_qa/politeness_rewriting_7000.jsonl'
else:
    file_path = os.path.join(args.root_path, args.dataset, args.linguistics, f'politeness_rewriting_{args.split}_7000-9000.jsonl')
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df_sampled_final.to_json(file_path, orient='records', lines=True)
print('Done!')