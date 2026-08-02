#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --job-name=cad-embed-cmp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Task 3 comparison: geometric-AE embedding vs Qwen3-Embedding (text) vs random,
# correlated against geometric ground-truth distance.
module load cuda/11.8
module load anaconda
source activate cad

cd $HOME/cad-copilot
mkdir -p results
python scripts/embed_compare.py \
    --dataset dataset \
    --ae-ckpt checkpoints/cad_ae.pt \
    --n-designs 300 --n-pairs 3000 \
    --out results/embed_compare.json
