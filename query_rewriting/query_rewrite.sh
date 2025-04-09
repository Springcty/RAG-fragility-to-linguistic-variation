#!/bin/bash
#SBATCH --job-name=rewrites_all
#SBATCH --output=logs/%x_%j_%a.log
#SBATCH --error=logs/testlogs/%x_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --gres=gpu:1              
#SBATCH --time=20:00:00            
#SBATCH --array=0-8             

combos=(
    "ms_marco formality"
    "ms_marco politeness"
    "EntityQuestions formality"
    "EntityQuestions politeness"
    "natural_questions formality"
    "natural_questions politeness"
    "PopQA formality"
    "PopQA politeness"
    "ms_marco formality"
)

combo=${combos[$SLURM_ARRAY_TASK_ID]}
dataset=$(echo $combo | cut -d' ' -f1)
mod=$(echo $combo | cut -d' ' -f2)
echo "Running for dataset: $dataset, modification: $mod"



csv_path="./data/${dataset}/${mod}/top_10000.csv" #Filter out top 10000 queries for the required linguistic variant per dataset


if [ "$dataset" == "EntityQuestions" ]; then
    dataset_out="entity_questions"
elif [ "$dataset" == "PopQA" ]; then
    dataset_out="popqa"
elif [ "$dataset" == "HotPotQA" ]; then
    dataset_out="hotpot_qa"
else
    dataset_out=$dataset
fi


output_path="./result/${dataset_out}/${mod}/final.csv"

# Debug: print the current settings
echo "Running for dataset: $dataset, modification: $mod"
echo "Input CSV: $csv_path"
echo "Output CSV: $output_path"

# Run the Python script with the corresponding parameters
python ./query_rewriting/transform_together_prompt.py \
    --big_csv_path "$csv_path" \
    --output_final_csv "$output_path" \
    --modification "$mod" \
    --model "gpt-4o-mini" \