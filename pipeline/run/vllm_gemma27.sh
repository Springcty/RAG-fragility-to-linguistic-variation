#!/bin/sh 
#SBATCH --job-name=gemma27
#SBATCH --error=logs/gemma27.%j.err
#SBATCH --output=logs/gemma27.%j.out
#SBATCH --gres=gpu:A6000:2
#SBATCH --partition=general
#SBATCH --mem=96Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm_server

echo "google/gemma-2-27b-it"
export VLLM_LOGGING_LEVEL=ERROR
echo "Port: 8080"

vllm serve google/gemma-2-27b-it \
    --download-dir /data/user_data/tianyuca/models/ \
    --dtype auto \
    --port 8080 \
    --device cuda \
    --tensor-parallel-size 2 \
    --max-model-len 4096