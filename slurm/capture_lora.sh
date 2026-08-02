#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=05:00:00
#SBATCH --job-name=cad-capture-lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Capture LoRA predicted curves on a 200-image subset (first 200 of the eval split),
# saving them for the CPU-side symbolic refinement stage. Only GPU step of Stage 2.
module load cuda/11.8
module load anaconda
source activate cad

cd $HOME/cad-copilot
mkdir -p results
python scripts/eval_model.py \
    --dataset dataset \
    --adapter checkpoints/lora-qwen3vl-2b \
    --metric chamfer \
    --n-eval 500 --limit 200 \
    --max-new-tokens 1536 \
    --out results/eval_lora_subset200.jsonl \
    --dump-pred results/pred_lora_subset200.jsonl
