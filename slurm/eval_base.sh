#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --job-name=cad-eval-base
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Baseline evaluation of the stock Qwen3-VL-2B on CAD sketch parsing.
module load cuda/11.8
module load anaconda
source activate cad

cd $HOME/cad-copilot
python scripts/eval_model.py --dataset dataset --metric chamfer --out eval_base_chamfer.jsonl
