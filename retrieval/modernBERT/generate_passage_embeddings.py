# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import bz2
import json
import csv
import argparse
import pickle
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import src.slurm
import src.normalize_text

rank = os.getenv('SLURM_PROCID')
print('-'*10, f'Rank: {rank}', '-'*10)


def load_passages(path):
    print(f"Loading passages from: {path}")
    passages = []
    
    if 'psgs_w100' in path:
        if not os.path.exists(path):
            print(f"{path} does not exist")
            return passages
        with open(path) as fin:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)

    elif 'Tevatron/msmarco-passage-corpus' in path:
        ds = load_dataset("Tevatron/msmarco-passage-corpus")
        passages = ds['train'].rename_column('docid', 'id').to_list()
    
    print(f"Loaded {len(passages)} passages.")
    return passages


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    

def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    
    with torch.no_grad():
        for k, p in enumerate(tqdm(passages)):
            batch_ids.append(p["id"])
            text = p["title"] + " " + p["text"]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            text = 'search_document: ' + text
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                ).to(args.device)

                # forward pass
                outputs = model(**encoded_batch)
                
                # mean pool and normalize
                embeddings_batch = mean_pooling(outputs, encoded_batch['attention_mask'])
                embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)
                
                # move to CPU and store
                allembeddings.append(embeddings_batch.cpu())
                total += len(batch_ids)
                allids.extend(batch_ids)
                
                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f'Encoding documents {total}')

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def main(args):
    # Load model and tokenizer
    model_name = 'nomic-ai/modernbert-embed-base'
    print(f'Loading model {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Prepare device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(args.device)
    model.eval()

    # Load passages
    if args.corpus == 'ms_marco':
        args.passages_path = 'Tevatron/msmarco-passage-corpus'
        args.output_dir = os.path.join(args.output_dir, 'ms_marco')
    elif args.corpus == 'wikipedia':
        args.passages_path = os.path.join(args.passages_path, 'psgs_w100.tsv')
        args.output_dir = os.path.join(args.output_dir, 'wikipedia_embeddings')
    
    passages = load_passages(args.passages_path)
    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    print(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.")

    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    save_file = os.path.join(args.output_dir, f"passages_{args.shard_id:02d}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(allids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.")

    # Finalize distributed mode
    src.slurm.finalize_distributed_mode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages_path", type=str,
                        default='/data/QL_dataset', help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default="/data/QL_dataset", help="dir path to save embeddings")
    parser.add_argument("--corpus", type=str, help="corpus name")
    
    parser.add_argument("--shard_id", type=int, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training, set by SLURM')
    parser.add_argument('--main_port', type=int, default=12345, help='main port for distributed training, set by SLURM')
    
    args = parser.parse_args()

    src.slurm.init_distributed_mode(args)
    print(args)

    main(args)
