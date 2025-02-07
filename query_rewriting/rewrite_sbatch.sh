#!/bin/bash
#SBATCH --job-name=query_rewrite
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-5     # Adjust based on total job count
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G          # Memory allocation (adjust if needed)
#SBATCH --gres=gpu:A6000:1        # Request 1 GPU

# Define arrays for your parameters.
DATASETS=("ms_marco" "EntityQuestions")
# DATASETS=("EntityQuestions")
MODIFICATIONS=("politeness")
PROMPT_TYPES=("prompt_1" "prompt_2" "prompt_3")
# PROMPT_TYPES=("prompt_3")
# Calculate total number of combinations.
TOTAL_COMBINATIONS=$((${#DATASETS[@]} * ${#MODIFICATIONS[@]} * ${#PROMPT_TYPES[@]}))

# Determine indices for the current job.
INDEX=$SLURM_ARRAY_TASK_ID
NUM_PROMPTS=${#PROMPT_TYPES[@]}
NUM_MODS=${#MODIFICATIONS[@]}

# Compute indices (using integer arithmetic):
dataset_index=$(( INDEX / (NUM_MODS * NUM_PROMPTS) ))
mod_index=$(( (INDEX / NUM_PROMPTS) % NUM_MODS ))
prompt_index=$(( INDEX % NUM_PROMPTS ))

# Get the actual parameter values:
DATASET=${DATASETS[$dataset_index]}
MODIFICATION=${MODIFICATIONS[$mod_index]}
PROMPT_TYPE=${PROMPT_TYPES[$prompt_index]}

# Define other parameters like sample_size.
SAMPLE_SIZE=10000

# Create the output directory in the desired structure.
OUTPUT_DIR="data/user_data/tianyuca/QL_dataset/${DATASET}/${MODIFICATION}"
mkdir -p "${OUTPUT_DIR}"

# Define output file paths.
OUTPUT_MOD="${OUTPUT_DIR}/${PROMPT_TYPE}_modified.csv"
OUTPUT_OG="${OUTPUT_DIR}/${PROMPT_TYPE}_original.csv"

echo "Running combination:"
echo "  Dataset: $DATASET"
echo "  Modification: $MODIFICATION"
echo "  Prompt Type: $PROMPT_TYPE"
echo "  Output MOD: $OUTPUT_MOD"

# Run your Python script with the corresponding parameters.
python query_rewriting/test_rewrite_prompts.py \
    --dataset "$DATASET" \
    --sample_size $SAMPLE_SIZE \
    --modification "$MODIFICATION" \
    --output_og "$OUTPUT_OG" \
    --output_mod "$OUTPUT_MOD" \
    --prompt_type "$PROMPT_TYPE" \
    # --full_sampling \