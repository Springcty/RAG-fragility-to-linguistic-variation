import argparse
import os
import pandas as pd

from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
from utils.few_shot_prompting import few_2shot_examples
from utils.vllm_inference import vllm_inference


parser = argparse.ArgumentParser(description='RAG pipeline')
parser.add_argument('--data_path', type=str, default='/data/QL_result/gpt-4o-mini',
                    help='The root path to load the retrieval results')
parser.add_argument('--retrieval', type=str, default='ModernBERT', 
                    help='The retrieval method from ["ModernBERT", "contriever"]')
parser.add_argument('--dataset', type=str, default='popqa',
                    help='Name of the QA dataset from ["popqa", "entity_questions" "ms_marco" "natural_questions"]')
parser.add_argument('--linguistics', type=str, default='formality',
                    help='The linguistic properties of the query to be modified, from["readability" "back_translated" "edited_query_char" "formality" "politeness"]')
parser.add_argument('--modified', type=str, default='original',
                    help='The type of query to be modified, from ["original", "modified"]')


# retrieval
parser.add_argument('--n_docs', type=int, default=5,
                    help='Number of documents to retrieve')

# vllm generation
parser.add_argument('--vllm_url', type=str, default='http://babel-X-X:9010/v1',
                    help='The URL of the vLLM server')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                    help='LLM used for generation')
parser.add_argument('--temperature', type=float, default=0.5,
                    help='The temperature for generation')
parser.add_argument('--max_tokens', type=int, default=128, # 64 for gpt-3.5-turbo rewriting, 128 for gpt-4o-mini
                    help='The maximum tokens for generation')
parser.add_argument('--top_p', type=float, default=0.90,
                    help='The top p for generation')
parser.add_argument('--requests_per_minute', type=int, default=150,
                    help='The requests per minute for generation')
parser.add_argument('--num_responses_per_prompt', type=int, default=1,
                    help='The number of responses per prompt for generation')


args = parser.parse_args()
print(args)


def combine_texts(ctxs_list):
    return '\n\n'.join(ctx['text'] for ctx in ctxs_list)


# Set prompt template for RAG with few-shot examples
def format_rag_prompt(data_df):
    system_prompt_template = '''You are a professional question-answer task assistant. Use the following pieces of retrieved context to answer the question briefly. 

Context: 
{contexts}

Below are examples of questions and answers:
{few_shot_examples}

Now, it's your turn to answer the question below. The answer should contain ONLY one sentence and DO NOT explain reasons.
'''
    user_prompt_template = 'Question: {question}\nAnswer:'
    
    # Generate prompt list for vLLM generation
    data_df['prompt'] = data_df.apply(
        lambda row: (
            system_prompt_template.format(
                contexts=row['combined_text'],
                few_shot_examples=few_2shot_examples[args.dataset][args.linguistics],
            ),
            user_prompt_template.format(question=row['question'])
        ),
        axis=1,
    )
    return data_df['prompt'].tolist()


def offline_generation(prompts):
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.90,
        max_tokens=64,
    )
    llm = LLM(model=args.model_name, download_dir='/data/user_data/tianyuca/models')
    completion = llm.generate(prompts, sampling_params)
    return completion


def main():
    data_path = os.path.join(args.data_path, args.dataset, args.linguistics, args.retrieval)
    result_path = os.path.join(data_path, args.model_name.split('/')[1])
    
    if os.path.exists(f'{result_path}/{args.modified}_generation.jsonl'):
        print(f'Generation results already exist in {result_path}/{args.modified}_generation.jsonl')
        return
    
    # Load the dataset with retrievals
    file_name = os.path.join(data_path, f'{args.modified}_retrieval.jsonl')
    print(f'Loading queries and retrieval results from {file_name}')
    data_df = pd.read_json(file_name, lines=True)

    # Generate context from several retrieval documents
    print('Creating combined retrieval context...')
    data_df['ctxs'] = data_df['ctxs'].apply(lambda x: x[:args.n_docs])
    data_df['combined_text'] = data_df['ctxs'].apply(combine_texts)
    
    # Set prompt template for RAG with few-shot examples
    print('Formatting prompts...')
    prompts = format_rag_prompt(data_df)
    
    print('Generating responses using vllm...')
    # OpenAI Compatible Server
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.vllm_url,
    )
    results = vllm_inference(
        client=client,
        prompts=prompts,
        model=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        requests_per_minute=args.requests_per_minute,
        num_responses_per_prompt=args.num_responses_per_prompt,
    )
    
    # Store the generation results
    print('Storing the generation results...')
    data_df['generation'] = results
    os.makedirs(result_path, exist_ok=True)
    data_df.to_json(f'{result_path}/{args.modified}_generation.jsonl', orient='records', lines=True)
    print(f'Generation results are stored in {result_path}/{args.modified}_generation.jsonl')
    

if __name__ == '__main__':
    main()