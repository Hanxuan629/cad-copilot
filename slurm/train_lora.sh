#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=4
#SBATCH --time=06:00:00
#SBATCH --job-name=cad-train-lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# LoRA fine-tune Qwen3-VL-2B. Wall limit is 6h (QoS max); the script checkpoints
# every 100 steps and auto-resumes, so if this is cut off just resubmit the SAME job:
#   sbatch slurm/train_lora.sh    # picks up from the latest checkpoint
module load cuda/11.8
module load anaconda
source activate cad

cd $HOME/cad-copilot
python scripts/train_lora.py \
    --dataset dataset \
    --output checkpoints/lora-qwen3vl-2b \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 1e-4 \
    --save-steps 100
