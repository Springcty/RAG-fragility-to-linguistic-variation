#!/bin/sh 
#SBATCH --job-name=qwen72
#SBATCH --error=logs/qwen72.%j.err
#SBATCH --output=logs/qwen72.%j.out
#SBATCH --gres=gpu:A6000:4
#SBATCH --partition=general
#SBATCH --mem=192Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm_server

echo "Qwen/Qwen2.5-72B-Instruct"
export VLLM_LOGGING_LEVEL=ERROR
echo "Port: 8090"

vllm serve Qwen/Qwen2.5-72B-Instruct \
    --download-dir /data/user_data/tianyuca/models/ \
    --dtype auto \
    --port 8090 \
    --device cuda \
    --tensor-parallel-size 4 \
    --max-model-len 4096
