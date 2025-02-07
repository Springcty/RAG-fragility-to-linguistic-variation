#!/bin/bash
#SBATCH --job-name=rewrites_all
#SBATCH --output=logs/rewrites_all_%A_%a.log
#SBATCH --error=logs/rewrites_all_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1              # 1 GPU per task
#SBATCH --time=12:00:00            # Adjust time as needed
#SBATCH --array=0-5           # 36 tasks total, run 5 in parallel

# (Optional) Load modules and activate environment
module load python
source ~/your_env/bin/activate  # Or conda activate your_env

#------------------------------------------------------------------------------
# DEFINE OUR 4x3x3 = 36 COMBOS
#------------------------------------------------------------------------------
declare -a combos=(
  # Dataset: PopQA
#   "PopQA formality prompt_1"
#   "PopQA formality prompt_2"
#   "PopQA formality prompt_3"
#   "PopQA readability prompt_1"
#   "PopQA readability prompt_2"
#   "PopQA readability prompt_3"
#   "PopQA politeness prompt_1"
#   "PopQA politeness prompt_2"
#   "PopQA politeness prompt_3"

#   # Dataset: natural_questions
#   "natural_questions formality prompt_1"
#   "natural_questions formality prompt_2"
#   "natural_questions formality prompt_3"
#   "natural_questions readability prompt_1"
#   "natural_questions readability prompt_2"
  "natural_questions readability prompt_3"
#   "natural_questions politeness prompt_1"
#   "natural_questions politeness prompt_2"
#   "natural_questions politeness prompt_3"

#   # Dataset: ms_marco
  # "ms_marco formality prompt_1"
  # "ms_marco formality prompt_2"
  # "ms_marco formality prompt_3"
  "ms_marco readability prompt_1"
  # "ms_marco readability prompt_2"
  "ms_marco readability prompt_3"
  # "ms_marco politeness prompt_1"
  # "ms_marco politeness prompt_2"
  # "ms_marco politeness prompt_3"

  # # Dataset: EntityQuestions
  # "EntityQuestions formality prompt_1"
  # "EntityQuestions formality prompt_2"
  # "EntityQuestions formality prompt_3"
#   "EntityQuestions readability prompt_1"
#   "EntityQuestions readability prompt_2"
  "EntityQuestions readability prompt_3"
#   "EntityQuestions politeness prompt_1"
#   "EntityQuestions politeness prompt_2"
#   "EntityQuestions politeness prompt_3"
# )

#------------------------------------------------------------------------------
# PARSE CURRENT ARRAY TASK
#------------------------------------------------------------------------------
combo="${combos[$SLURM_ARRAY_TASK_ID]}"
dataset=$(echo "$combo" | awk '{print $1}')
modification=$(echo "$combo" | awk '{print $2}')
prompt_type=$(echo "$combo" | awk '{print $3}')

echo "===================================================="
echo " SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo " Dataset:            $dataset"
echo " Modification:       $modification"
echo " Prompt Type:        $prompt_type"
echo "===================================================="

#------------------------------------------------------------------------------
# BUILD PATHS
#------------------------------------------------------------------------------
# 1) Path to top_5k CSV
TOP5K_PATH="data/user_data/neelbhan/top_5k/${dataset}/${modification}/top_5000.csv"

# 2) Path to modified CSV (based on prompt_type)
MODIFIED_CSV="data/user_data/tianyuca/QL_dataset/${dataset}/${modification}/${prompt_type}_modified.csv"

# 3) Final output CSV
OUTPUT_FINAL="data/user_data/tianyuca/QL_dataset/${dataset}/${modification}/${prompt_type}_final_df.csv"

# Create directories if needed (just in case)
mkdir -p "$(dirname "$OUTPUT_FINAL")"
mkdir -p "logs"

#------------------------------------------------------------------------------
# CHECK IF MODIFIED CSV EXISTS
#------------------------------------------------------------------------------
if [[ -f "$MODIFIED_CSV" ]]; then
    echo "Modified CSV found: $MODIFIED_CSV"
    INPUT_FILE="$MODIFIED_CSV"
else
    echo "Modified CSV missing. Running on top_5k instead: $TOP5K_PATH"
    INPUT_FILE="$TOP5K_PATH"
fi

#------------------------------------------------------------------------------
# RUN THE SCRIPT
#------------------------------------------------------------------------------
python query_rewriting/non_candidates.py \
  --top5k_path "${TOP5K_PATH}" \
  --modified_csv_path "${INPUT_FILE}" \
  --output_final_csv "${OUTPUT_FINAL}" \
  --max_retries 5 \
  --prompt_type "${prompt_type}" \
  --modification "${modification}" \
  --model "gpt-3.5-turbo" \
  --temperature 1.0 \
  --max_response_tokens 100 \
  --top_p 1.0 \
  --requests_per_minute 150

#------------------------------------------------------------------------------
# DONE
#------------------------------------------------------------------------------
echo "Finished: [Dataset: $dataset] [Modification: $modification] [Prompt: $prompt_type]"