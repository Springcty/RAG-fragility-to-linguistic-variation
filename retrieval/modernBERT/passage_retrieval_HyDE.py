# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import csv
import json
import pickle
import time
import glob
import ast
import logging
import bz2

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import src.index
import src.slurm
from src.evaluation import calculate_matches
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    

def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            q = 'search_query: ' + q
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                # tokenize
                encoded_batch = tokenizer(
                    batch_question,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(args.device)
                
                # forward pass
                outputs = model(**encoded_batch)
                
                # mean pool and normalize
                embeddings_batch = mean_pooling(outputs, encoded_batch['attention_mask'])
                embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)
                
                embeddings.append(embeddings_batch.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    logging.info(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        logging.info(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    logging.info("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logging.info(f"Validation results: top k documents hits {top_k_hits}")
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    logging.info(message)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    data = []
    if not os.path.exists(data_path):
        data_path = data_path.replace("_queries.jsonl", ".jsonl")

    with open(data_path, "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)

            if type(example["answers"]) == str:
                example["answers"] = ast.literal_eval(example["answers"])

            if 'hyde' in data_path:
                example_hyde = {'id': example['id'], 'question': example['question'], 'answers': example['answers']}
                query = example['question']
                hyde_documents = example['hyde_docs']
                for i, doc in enumerate(hyde_documents):
                    query_hyde = query + ' ' + doc
                    example_hyde[f'query_hyde_{i}'] = query_hyde
                data.append(example_hyde)
            else:
                data.append(example)
    
    return data


def load_passages(path):
    if not os.path.exists(path):
        logging.info(f"{path} does not exist")
        return
    passages = []
    
    if 'psgs_w100' in path:
        logging.info(f"Loading passages from: {path}")
        with open(path) as fin:
            reader = csv.reader(fin, delimiter='\t')
            for k, row in enumerate(reader):
                if not row[0] == 'id':
                    ex = {'id': row[0], 'title': row[2], 'text': row[1]}
                    passages.append(ex)
    elif 'ms_marco' in path:
        logging.info(f"Loading passages from: Tevatron/msmarco-passage-corpus")
        ds = load_dataset('Tevatron/msmarco-passage-corpus')
        passages = ds['train'].rename_column('docid', 'id').to_list()
    elif 'enwiki-20171001-pages-meta-current-withlinks-abstracts' in path:
        # {path}/AA/wiki_00.bz2
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.bz2'):
                    with bz2.open(os.path.join(root, file), 'rt') as f:
                        for line in f:
                            line = json.loads(line)
                            ex = {'id': line['id'], 'title': line['title'], 'text': ''.join(line['text'])}
                            passages.append(ex)
    
    print(f"Loaded {len(passages)} passages.")
    return passages


def main(args):
    # Set up loading and saving directories
    # args.output_dir = '/data/group_data/maartens_lab_miis24/HyDE
    log_filepath = os.path.join(args.output_dir, args.dataset, args.linguistic, "ModernBERT", f"{args.modified}_retrieval.log")
    
    if args.dataset == 'ms_marco':
        passages_embeddings_path = os.path.join(args.passages_path, 'ms_marco', 'passages_*')
        passages_path = os.path.join(args.passages_path, 'ms_marco')
    else:
        # args.passages_path = '/data/user_data/tianyuca/QL_dataset
        passages_embeddings_path = os.path.join(args.passages_path, 'wikipedia_embeddings', 'passages_*')
        passages_path = os.path.join(args.passages_path, 'psgs_w100.tsv')
    
    # args.data_path = '/data/group_data/maartens_lab_miis24/hyde_retrieval_1_rewrite_docs
    data_path = os.path.join(args.data_path, args.dataset, args.linguistic, f'{args.modified}_queries.jsonl')
    output_path = os.path.join(args.output_dir, args.dataset, args.linguistic, "ModernBERT", f"{args.modified}_retrieval.jsonl")

    # Set up logging
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    logging.basicConfig(filename=log_filepath, filemode='w', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Arguments: {args}")
    
    # Load model
    logging.info(f'Loading model {args.model_name}') # model_name = 'nomic-ai/modernbert-embed-base'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    # Prepare device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(args.device)
    model.eval()

    # Build a FAISS index
    index = src.index.Indexer(args.embedding_dim, args.m, args.nbits)

    # index all passages
    input_paths = glob.glob(passages_embeddings_path)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        logging.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        logging.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    # load passages
    passages = load_passages(passages_path)
    passage_id_map = {x["id"]: x for x in passages}
    
    # load queries
    data = load_data(data_path)
    
    # encode queries
    queries_hyde_embeddings = []
    for i in range(4):
        queries_hyde = [ex[f'query_hyde_{i}'] for ex in data]
        embedding = embed_queries(args, queries_hyde, model, tokenizer)
        queries_hyde_embeddings.append(embedding)
    # Calculate the mean of the 4 query embeddings
    questions_embedding = np.mean(queries_hyde_embeddings, axis=0)

    # get top k results
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
    logging.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
    add_passages(data, passage_id_map, top_ids_and_scores)
    
    # evaluate results: has answer
    hasanswer = validate(data, args.validation_workers)
    add_hasanswer(data, hasanswer)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for ex in data:
            json.dump(ex, fout, ensure_ascii=False)
            fout.write("\n")
    logging.info(f"Saved results to {output_path}")

'''
data_path
dataset
linguistic
modified
passage_path
passage_embeddings_path

output_path
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    '''
    python passage_retrieval.py --dataset popqa --linguistic readability --modified original --n_docs 5 --lowercase --normalize_text --save_or_load_index \
    --nlist 16384 --m 64 --nbits 8 
    '''
    # General experimental args
    parser.add_argument("--passages_path", type=str, 
                        default='/data/user_data/tianyuca/QL_dataset', help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, 
                        default='/data/group_data/maartens_lab_miis24/QL_result', help="Results are written to outputdir with data suffix")
    parser.add_argument("--data_path", type=str,
                        default='/data/group_data/maartens_lab_miis24/QL_dataset', help='path to QA dataset')
    parser.add_argument("--rewriting_model", type=str,
                        default='gpt-4o-mini', help='model used to generate modified queries')
    parser.add_argument('--dataset', type=str,
                        default='popqa', help='Name of the QA dataset, including ["popqa", "entity_questions", "ms_marco", "natural_questions"]')
    parser.add_argument('--linguistic', type=str,
                        default=None, help='linguistic characteristics to be analyzed, including ["readability", "formality", "politeness", "grammatical_correctness"]')
    parser.add_argument('--modified', type=str,
                        default='original', help='the original query or the modified query')
    
    # Retrieval args
    parser.add_argument("--n_docs", type=int, 
                        default=100, help="Number of documents to retrieve per questions")
    parser.add_argument("--validation_workers", type=int, 
                        default=32, help="Number of parallel processes to validate results")

    # Question encoding args
    parser.add_argument("--per_gpu_batch_size", type=int, 
                        default=64, help="Batch size for question encoding")
    parser.add_argument("--model_name", type=str, 
                        default='nomic-ai/modernbert-embed-base', help="path to directory containing model weights and config file")
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    
    # Indexing args
    parser.add_argument("--save_or_load_index", action="store_true", 
                        help="If enabled, save index and load index if it exists")
    parser.add_argument("--indexing_batch_size", type=int, 
                        default=1000000, help="Batch size of the number of passages indexed")
    parser.add_argument("--embedding_dim", type=int, 
                        default=768, help='embedding_dim = all_embeddings.shape[1]')
    # parser.add_argument('--nlist', type=int, default=16384,
    #                     help='number of centroid clusters')
    parser.add_argument("--m", type=int,
                        default=64, help="Number of subquantizer used for vector quantization, if 0 flat index is used",)
    parser.add_argument("--nbits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")

    args = parser.parse_args()
    print(args)
    
    src.slurm.init_distributed_mode(args)
    main(args)
