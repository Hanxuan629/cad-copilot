#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=05:00:00
#SBATCH --job-name=cad-eval-lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Evaluate the LoRA-adapted model, for direct comparison against eval_base.
module load cuda/11.8
module load anaconda
source activate cad

cd $HOME/cad-copilot
python scripts/eval_model.py \
    --dataset dataset \
    --adapter checkpoints/lora-qwen3vl-2b \
    --metric chamfer \
    --max-new-tokens 1536 \
    --out eval_lora_chamfer.jsonl
