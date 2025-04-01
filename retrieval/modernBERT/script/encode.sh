#!/bin/sh 
#SBATCH --job-name=gen_embeddings
#SBATCH --output=logs/gen_embeddings.%j.out
#SBATCH --error=logs/gen_embeddings.%j.err
#SBATCH --ntasks=6
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48Gb
#SBATCH --partition=general
#SBATCH -t 2-00:00:00

export NCCL_P2P_DISABLE=1
source ~/miniconda3/etc/profile.d/conda.sh
source /usr/share/Modules/init/bash
module load cuda-12.5
conda activate vllm

# Initialize main_port
MAIN_PORT=12345
 
srun python ../generate_passage_embeddings.py \
    --main_port $MAIN_PORT \
    --shard_id $SLURM_PROCID \
    --num_shards $SLURM_NTASKS \
    --lowercase \
    --normalize_text \