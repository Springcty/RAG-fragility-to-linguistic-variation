#!/bin/sh 
#SBATCH --job-name=ma-rerank-retrieval
#SBATCH --output=logs/ma-rerank-retrieval.%j.out
#SBATCH --error=logs/ma-rerank-retrieval.%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48Gb
#SBATCH --partition=general
#SBATCH -t 2-00:00:00

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
module load cuda-12.5
conda activate vllm

dataset="ms_marco"
# linguistics=("readability" "politeness")
linguistics=("back_translated" "edited_query_char" "formality")
modifications=("modified" "original")

for linguistic in "${linguistics[@]}"; do
    for modified in "${modifications[@]}"; do
        python ../passage_reranking.py \
            --dataset "$dataset" --linguistic "$linguistic" --modified "$modified"
    done
done