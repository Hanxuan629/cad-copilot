#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --job-name=cad-ae
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Task 3: CAD-design autoencoder -> embedding -> embedding-distance similarity metric.
# Pure geometry (no VLM). Trains on the rendered dataset's labels.
module load cuda/11.8
module load anaconda
source activate cad

cd $HOME/cad-copilot
python scripts/cad_autoencoder.py \
    --dataset dataset \
    --emb 256 \
    --epochs 40 \
    --batch-size 128 \
    --out checkpoints/cad_ae.pt
