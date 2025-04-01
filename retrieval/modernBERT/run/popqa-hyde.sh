#!/bin/sh 
#SBATCH --job-name=popqa-hyde-retrieval
#SBATCH --output=logs/popqa-hyde-retrieval.%j.out
#SBATCH --error=logs/popqa-hyde-retrieval.%j.err
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

dataset="popqa"
# linguistics=("readability" "back_translated" "edited_query_char" "politeness")
linguistics=("formality" )
modifications=("original") #  "modified"

for linguistic in "${linguistics[@]}"; do
    for modified in "${modifications[@]}"; do
        python ../passage_retrieval_HyDE.py \
            --output_dir /data/group_data/maartens_lab_miis24/HyDE --data_path /data/group_data/maartens_lab_miis24/hyde_retrieval_1_rewrite_docs \
            --dataset "$dataset" --linguistic "$linguistic" --modified "$modified" \
            --n_docs 100 \
            --lowercase --normalize_text \
            --save_or_load_index \
            --m 64 --nbits 8
    done
done