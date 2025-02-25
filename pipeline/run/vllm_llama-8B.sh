#!/bin/sh 
#SBATCH --job-name=ll3-1-8b-2
#SBATCH --error=logs/ll3-1-8b-2.%j.err
#SBATCH --output=logs/ll3-1-8b-2.%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=48Gb
#SBATCH -t 2-00:00:00              # time limit: (D-HH:MM) 

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
# module load cuda-12.1
conda activate vllm_server
 
echo "Llama-3.1-8B-Instruct"
export VLLM_LOGGING_LEVEL=ERROR
echo "Port: 9010"

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --download-dir /data/user_data/tianyuca/models/ \
    --dtype auto \
    --port 9010 \
    --device cuda \
    --tensor-parallel-size 1