#!/bin/bash
#SBATCH --job-name=readability_rewrites
#SBATCH --output=logs/%x_%j_%a.log
#SBATCH --error=logs/testlogs/%x_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --gres=gpu:1              
#SBATCH --time=20:00:00            
#SBATCH --array=0-8             

datasets=(
    "ms_marco"
    "entity_questions"
    "natural_questions"
    "popqa"
)

# Available splits for each dataset
    # ms_marco: "validation" "test"
    # natural_questions: "validation" "test"
    # popqa: "validation"
    # entity_questions: "validation"
split="validation"

for dataset in "${datasets[@]}"; do
    python readability_rewriting.py --root_path ./data \
    --dataset "$dataset" --split "$split" \
    --linguistics readability \
    --model gpt-4o-mini
done