#!/bin/bash
#SBATCH --gres=gpu:A6000:1         # Request 1 GPU
#SBATCH --partition=general        # Adjust as needed for your cluster
#SBATCH --mem=150Gb                # Request 150 GB memory
#SBATCH --cpus-per-task=8          # Allocate sufficient CPUs
#SBATCH -t 2-00:00:00              # Set the time limit (2 days)
#SBATCH --job-name=retrieval
#SBATCH --error=/home/neelbhan/QueryLinguistic/logs/retrieval.%j.err
#SBATCH --output=/home/neelbhan/QueryLinguistic/logs/retrieval.%j.out

# Disable NCCL P2P (for better performance in some multi-GPU setups)
cd ./data/contriever
export NCCL_P2P_DISABLE=1

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate halurag

# Load required modules
module load cuda-12.5

# Run the passage embedding generation script
python generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever \
    --output_dir ./data/contriever_embeddings  \
    --passages ./data/passages/msmarco_passage.jsonl \
    --shard_id 0 --num_shards 1