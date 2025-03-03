#!/bin/sh 
#SBATCH --job-name=qwen0_5
#SBATCH --error=logs/qwen0_5.%j.err
#SBATCH --output=logs/qwen0_5.%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=48Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm_server

echo "Qwen/Qwen2.5-0.5B-Instruct"
export VLLM_LOGGING_LEVEL=ERROR
echo "Port: 8024"

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --download-dir /data/user_data/tianyuca/models/ \
    --dtype auto \
    --port 8024 \
    --device cuda \
    --tensor-parallel-size 1 \
    --max-model-len 4096
