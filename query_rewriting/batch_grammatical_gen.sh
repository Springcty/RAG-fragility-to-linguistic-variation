#!/bin/bash
#SBATCH --job-name=grammar_mods
#SBATCH --output=logs/grammar_mods_%A_%a.out
#SBATCH --error=logs/grammar_mods_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 2-00:00:00
#SBATCH --array=0  # Update this based on the number of datasets you want to run


# Manually specify the datasets you want to run on
datasets=("PopQA" "EntityQuestions" "natural_questions" "ms_marco")  # Modify this list as needed
dataset_name=${datasets[$SLURM_ARRAY_TASK_ID]}

cmd="python query_rewriting/analysing_metrics_datasets/grammar_conversion.py --dataset_name $dataset_name --full_sampling --apply_char --apply_back --pivot_lang af --output_file ./result/final.csv"
echo "Running: $cmd"
eval $cmd


