#!/bin/sh 
#SBATCH --job-name=entity-wk-retrieval
#SBATCH --output=logs/entity-wk-retrieval.%j.out
#SBATCH --error=logs/entity-wk-retrieval.%j.err
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48Gb
#SBATCH --partition=general
#SBATCH -t 2-00:00:00

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
module load cuda-12.5
conda activate vllm

dataset="entity_questions"
linguistics=("readability" "back_translated" "edited_query_char" "formality" "politeness")
modifications=("modified" "original")

for linguistic in "${linguistics[@]}"; do
    for modified in "${modifications[@]}"; do
        # skip if linguistic is back_translated and modified is original
        if [ "$linguistic" == "back_translated" ] && [ "$modified" == "original" ]; then
            continue
        fi
        python ../passage_retrieval.py \
            --dataset "$dataset" --linguistic "$linguistic" --modified "$modified" \
            --n_docs 100 \
            --lowercase --normalize_text \
            --save_or_load_index \
            --m 64 --nbits 8
    done
done