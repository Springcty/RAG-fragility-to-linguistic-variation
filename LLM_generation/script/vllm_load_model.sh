#!/bin/sh 
#SBATCH --job-name=model_name
#SBATCH --error=logs/model_name.%j.err
#SBATCH --output=logs/model_name.%j.out
#SBATCH --gres=gpu:gpu_name:4
#SBATCH --partition=gpu_partition
#SBATCH --mem=192Gb
#SBATCH -t 2-00:00:00

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm_server

model_name="model_name"
port="port"

echo "Model name: $model_name"
echo "Port: $port"

vllm serve $model_name \
    --download-dir /data/models/ \
    --dtype auto \
    --port $port \
    --device cuda \
    --tensor-parallel-size 4 \
    --max-model-len 4096