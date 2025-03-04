#!/bin/sh 
#SBATCH --job-name=qwen72-preempt
#SBATCH --error=logs/qwen72-preempt.%j.err
#SBATCH --output=logs/qwen72-preempt.%j.out
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=preempt
#SBATCH --mem=192Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM)

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm_server

model="Qwen/Qwen2.5-72B-Instruct"
port="8092"

export VLLM_LOGGING_LEVEL=ERROR
echo $model
echo "Port: $port"

vllm serve "$model" \
    --download-dir /data/user_data/tianyuca/models/ \
    --dtype auto \
    --port "$port" \
    --device cuda \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --enable_chunked_prefill True \
    --max_num_batched_tokens 8192
