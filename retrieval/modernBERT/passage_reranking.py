import os
import json
import argparse
import logging

import pandas as pd
from sentence_transformers import CrossEncoder, util

from passage_retrieval import validate

parser = argparse.ArgumentParser(description='RAG pipeline')
parser.add_argument('--data_path', type=str, default='/data/QL_result/gpt-4o-mini',
                    help='The root path to load the retrieval results')
parser.add_argument('--save_path', type=str, default='/data/Reranking',
                    help='The root path to save the reranking results')
parser.add_argument('--retrieval', type=str, default='ModernBERT', 
                    help='The retrieval method from ["ModernBERT", "contriever"]')
parser.add_argument('--dataset', type=str, default='popqa',
                    help='Name of the QA dataset from ["popqa", "entity_questions" "ms_marco" "natural_questions"]')
parser.add_argument('--linguistics', type=str, default='formality',
                    help='The linguistic properties of the query to be modified, from["readability" "back_translated" "edited_query_char" "formality" "politeness"]')
parser.add_argument('--modified', type=str, default='original',
                    help='The type of query to be modified, from ["original", "modified"]')
parser.add_argument("--validation_workers", type=int, 
                    default=32, help="Number of parallel processes to validate results")
args = parser.parse_args()
print(args)


def main():
    retrieval_path = os.path.join(args.data_path, args.dataset, args.linguistics, args.retrieval)
    reranking_path = os.path.join(args.save_path, args.dataset, args.linguistics, args.retrieval)
    os.makedirs(reranking_path, exist_ok=True)
    
    # Set up logging
    log_filepath = os.path.join(reranking_path, f'{args.modified}_retrieval.log')
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    logging.basicConfig(filename=log_filepath, filemode='w', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Arguments: {args}")
    
    # Load the retrieval results
    file_name = os.path.join(retrieval_path, f'{args.modified}_retrieval.jsonl')
    logging.info(f'Loading retrieval results from {file_name}')
    data_df = pd.read_json(file_name, orient='records', lines=True)
    data = data_df.to_dict(orient='records')
    
    # Reranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # iterate over the data
    for row in data:
        query = row['question']
        passages = row['ctxs']
        
        cross_input = [[query, passage['text']] for passage in passages]
        cross_scores = cross_encoder.predict(cross_input)
        
        # sort the passages based on the cross_scores
        sorted_passages = [passages[i] for i in cross_scores.argsort()[::-1]]
        row['ctxs'] = sorted_passages

    # Evaluate the reranking results
    logging.info('Evaluating the reranking results')
    hasanswer = validate(data, args.validation_workers)
    
    # Save the reranking results
    logging.info(f'Saving reranking results to {file_name}')
    file_name = os.path.join(reranking_path, f'{args.modified}_retrieval.jsonl')
    with open(file_name, "w") as fout:
        for ex in data:
            json.dump(ex, fout, ensure_ascii=False)
            fout.write("\n")

if __name__ == '__main__':
    main()