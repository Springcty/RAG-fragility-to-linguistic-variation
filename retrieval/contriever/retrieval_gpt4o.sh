#!/bin/bash
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=150Gb
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH --job-name=retrieval_ms_marco
#SBATCH --array=0-1  # 8 jobs * 5 modifications = 40 tasks
#SBATCH --error=logs/gpt_4o_retrieval/retrieval_%A_%a.err
#SBATCH --output=logs/gpt_4o_retrieval/retrieval_%A_%a.out

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate halurag
module load cuda-12.5


MODS=("formality" "politeness" "readability" "back_translated" "edited_query_char")



JOBS=(
    "entity_questions original_queries"
    "entity_questions modified_queries"
    "ms_marco original_queries"
    "ms_marco modified_queries"
    "natural_questions original_queries"
    "natural_questions modified_queries"
    "popqa original_queries"
    "popqa modified_queries"
)

n_mods=${#MODS[@]}

job_index=$(( SLURM_ARRAY_TASK_ID / n_mods ))
mod_index=$(( SLURM_ARRAY_TASK_ID % n_mods ))

JOB="${JOBS[$job_index]}"
set -- $JOB
DATASET=$1
FILE_TYPE=$2
MODIFICATION="${MODS[$mod_index]}"

DATA="./data/${DATASET}/${MODIFICATION}/${FILE_TYPE}.jsonl"
OUTPUT_DIR="./result/${DATASET}/${MODIFICATION}"

mkdir -p "$OUTPUT_DIR"

echo "Starting retrieval for $DATASET ($FILE_TYPE, modification: $MODIFICATION) on SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# Choose the appropriate retrieval command based on dataset
if [ "$DATASET" = "ms_marco" ]; then
    python /home/neelbhan/QueryLinguistic/retrieval/contriever/passage_retrieval.py \
        --model_name_or_path facebook/contriever \
        --passages ./data/ms_marco_retrieval_data/msmarco_passage.jsonl \
        --passages_embeddings "./data/contriever_embeddings/*" \
        --data "$DATA" \
        --output_dir "$OUTPUT_DIR" \
        --n_docs 100
else
    python /home/neelbhan/QueryLinguistic/retrieval/contriever/passage_retrieval.py \
        --model_name_or_path facebook/contriever \
        --passages ./data/psgs_w100.tsv \
        --passages_embeddings "./data/embeddings/wikipedia_embeddings/*" \
        --data "$DATA" \
        --output_dir "$OUTPUT_DIR" \
        --n_docs 100
fi

# ##################################### ${MODIFICATION^^} ###########################################