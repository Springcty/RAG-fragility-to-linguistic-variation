#!/bin/sh 
#SBATCH --job-name=gemma
#SBATCH --error=logs/gemma.%j.err
#SBATCH --output=logs/gemma.%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=48Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
conda activate vllm_server
 
echo "google/gemma-2-9b-it"
export VLLM_LOGGING_LEVEL=ERROR
echo "Port: 9010"

vllm serve google/gemma-2-9b-it \
    --download-dir /data/user_data/tianyuca/models/ \
    --dtype auto \
    --port 8080 \
    --device cuda \
    --tensor-parallel-size 1