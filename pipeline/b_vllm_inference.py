import argparse
import os
import pandas as pd

from vllm import LLM, SamplingParams
from openai import OpenAI
from utils import readability_2_shot, formality_2_shot

parser = argparse.ArgumentParser(description='RAG pipeline')
parser.add_argument('--data_path', type=str, default='/home/tianyuca/RAG/QueryLinguistic/retrieval_outputs',
                    help='The path to load the dataset')
parser.add_argument('--result_path', type=str, default='/data/user_data/tianyuca/QL_result',
                    help='The path to save the retrievel and generation results')
parser.add_argument('--dataset', type=str, default='popqa',
                    help='Name of the QA dataset from ["popqa", "entity"]')
parser.add_argument('--property', type=str, default='readability',
                    help='The linguistic properties of the query to be modified, from["readability", "formality", "concreteness"]')
parser.add_argument('--modified', type=str, default='modified')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                    help='LLM used for generation')

args = parser.parse_args()


def combine_texts(ctxs_list):
        return '\n\n'.join(ctx['text'] for ctx in ctxs_list)


def main():
    data_path = f'{args.data_path}/{args.dataset}/{args.property}'
    save_path = f'{args.result_path}/{args.dataset}/{args.property}'
    os.makedirs(save_path, exist_ok=True)
    
    # Load the dataset with retrievals
    data_df = pd.read_json(f'{data_path}/{args.modified}_metrics_qsampled.jsonl', lines=True)
    
    # Generate context from several retrieval documents
    data_df['combined_text'] = data_df['ctxs'].apply(combine_texts)
    
    # Set prompt template for RAG with few-shot examples
    prompt_template = '''You are a professional question-answer task assistant. Use the following pieces of retrieved context to answer the question briefly. 

Context: 
{contexts}

Below are examples of questions and answers:
{few_shot_examples}

Now, it's your turn to answer the question below. The answer should contain ONLY one sentence and DO NOT explain reasons.

Question: {question}
Answer:'''
    
    # Generate prompt list for vLLM generation
    data_df['prompt'] = data_df.apply(
        lambda row: prompt_template.format(
            contexts=row['combined_text'],
            few_shot_examples=readability_2_shot if args.property == 'readability' else formality_2_shot,
            question=row['question'],
        ),
        axis=1,
    )
    prompts = data_df['prompt'].tolist()
    
    # Initialize vLLM for generation
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.90,
        max_tokens=64,
    )
    llm = LLM(model=args.model_name)
    completion = llm.generate(prompts, sampling_params)
    
    # # OpenAI Compatible Server
    # client = OpenAI(
    #     base_url='https://babel-12-25:9010/v1',
    # )
    
    # completion = client.completions.create(
    #     model=args.model_name,
    #     prompt=prompts,
    #     temperature=0.5,
    #     top_p=0.90,
    #     max_tokens=64,
    # )
    
    # Store the generation results
    data_df['generation'] = completion # generated_text = output.outputs[0].text
    data_df.to_json(f'{save_path}/{args.modified}_generation.jsonl', orient='records', lines=True)


if __name__ == '__main__':
    main()