#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm

# Ensure that joblib uses only the allocated CPUs
export JOBLIB_NUM_THREADS=$SLURM_CPUS_PER_TASK

dataset=("popqa" "entity_questions" "natural_questions" "ms_marco")
linguistics=("back_translated" "edited_query_char" "formality" "politeness" "readability")
modified=("original" "modified")

model_name=("google/gemma-2-2b-it" "google/gemma-2-9b-it" "google/gemma-2-27b-it" "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-70B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-32B-Instruct" "Qwen/Qwen2.5-72B-Instruct")
retrieval=("ModernBERT")
# retrieval=("Contriever")
# retrieval=("none_retrieval")


for arg1 in "${dataset[@]}"; do
    for arg2 in "${linguistics[@]}"; do
        for arg3 in "${modified[@]}"; do
            for arg4 in "${retrieval[@]}"; do
                for arg5 in "${model_name[@]}"; do

                    echo "Evaluation, args: "$arg1", "$arg2", "$arg3", "$arg4", "$arg5""
                    python ../c_eval_generation.py \
                    --dataset "$arg1" --linguistics "$arg2" --modified "$arg3" \
                    --retrieval "$arg4" --model_name "$arg5"
                
                done
            done
        done
    done
done
