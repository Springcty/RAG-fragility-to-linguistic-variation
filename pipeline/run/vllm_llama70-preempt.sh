#!/bin/sh 
#SBATCH --job-name=ll3-1-70b-pr
#SBATCH --error=logs/ll3-1-70b-pr.%j.err
#SBATCH --output=logs/ll3-1-70b-pr.%j.out
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=preempt
#SBATCH --mem=192Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm_server

echo "Llama-3.1-70B-Instruct"
export VLLM_LOGGING_LEVEL=ERROR
echo "Port: 9010"

vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --download-dir /data/user_data/tianyuca/models/ \
    --dtype auto \
    --port 9010 \
    --device cuda \
    --tensor-parallel-size 4 \
    --max-model-len 4096
