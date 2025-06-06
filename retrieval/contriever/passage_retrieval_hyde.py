import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers
import openai  

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text
from pathlib import Path
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Generator:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self, prompt):
        return ""

class OpenAIGenerator(Generator):
    def __init__(self, model_name, api_key, base_url=None, n=4, max_tokens=256, temperature=0.7, top_p=1, 
                 frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name, api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success
        self.base_url = base_url
        self._client_init()
    
    @staticmethod
    def parse_response(response):
        to_return = []
        for g in response.choices:

            text = g.message.content
            to_return.append(text)
        return to_return
    
    def _client_init(self):
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def generate(self, prompt):
        get_results = False
        while not get_results:
            try:
                print("Generating result ......... ")
                result = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    max_completion_tokens=256,
                    temperature=self.temperature,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    top_p=self.top_p,
                    n=4, 
                    stop=self.stop
                    
                )
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)

class DummyGenerator:
    def __init__(self, n=8):
        self.n = n

    def generate(self, query):
        return [f"Hypothesis passage {i+1} for query: {query}" for i in range(self.n)]
    
def embed_hyde_queries(args, queries, model, tokenizer, generator):
    """
    For each query, generate multiple hypothesis passages, encode them along with the query,
    and aggregate the embeddings via mean pooling. Also stores query-hypothesis pairs in a JSONL file.
    """
    model.eval()
    all_embeddings = []

    output_dir = ".data/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)  
    output_jsonl_path = f"{output_dir}/generated_hyde_docs.jsonl"

    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        with torch.no_grad():
            for q in tqdm(queries, desc="Processing queries"):
                q_proc = q.lower() if args.lowercase else q
                if args.normalize_text:
                    q_proc = src.normalize_text.normalize(q_proc)
                prompt = f"Please write a passage to answer the question\nQuestion: {q_proc}\nPassage:"
                hyps = generator.generate(prompt)

                texts = [q_proc] + hyps

                encoded_batch = tokenizer.batch_encode_plus(
                    texts,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch) 

                agg_embedding = output.mean(dim=0, keepdim=True) 
                all_embeddings.append(agg_embedding.cpu())

                record = {
                    "query": q_proc,
                    "generated_docs": hyps
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"HyDE docs written to {output_jsonl_path}")

    return all_embeddings.numpy()

def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():
        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())
                batch_question = []
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.numpy()

def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")

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
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "

    return match_stats.questions_doc_hits

def add_passages(data, passages, top_passages_and_scores):

    merged_data = []

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
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]

def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data

def main(args):
    print(f"Loading model from: {args.model_name_or_path}")
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)


    passages = src.data.load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}

    data_paths = glob.glob(args.data)
    alldata = []
    for path in data_paths:
        data = load_data(path)
        output_path = os.path.join(args.output_dir, os.path.basename(path))
        queries = [ex["question"] for ex in data]
        
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            generator = OpenAIGenerator(
                model_name="gpt-4o-mini",
                api_key=openai_api_key,
                n=8,
                max_tokens=512,
                temperature=0.7,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=['\n\n\n'],
                wait_till_success=True
            )
        else:
            return -1
            generator = DummyGenerator(n=8)
        

        questions_embedding = embed_hyde_queries(args, queries, model, tokenizer, generator)
        
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        add_passages(data, passage_id_map, top_ids_and_scores)
        hasanswer = validate(data, args.validation_workers)
        add_hasanswer(data, hasanswer)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)
