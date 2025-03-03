#!/bin/sh 
#SBATCH --job-name=qwen14
#SBATCH --error=logs/qwen14.%j.err
#SBATCH --output=logs/qwen14.%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=48Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm_server

echo "Qwen/Qwen2.5-14B-Instruct"
export VLLM_LOGGING_LEVEL=ERROR
echo "Port: 8090"

# “auto” will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --download-dir /data/user_data/tianyuca/models/ \
    --dtype auto \
    --port 8090 \
    --device cuda \
    --tensor-parallel-size 1 \
    --max-model-len 4096
