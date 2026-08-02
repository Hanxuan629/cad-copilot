#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --job-name=cad-probe-match
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Zero-shot line-matching probe: stock Qwen3-VL-2B, text and visual prompts.
module load cuda/11.8
module load anaconda
source activate cad

cd $HOME/cad-copilot
mkdir -p results
python scripts/probe_match.py --data data/match_cad2tgt.jsonl --mode text \
    --limit 50 --out results/probe_text.jsonl
python scripts/probe_match.py --data data/match_cad2tgt.jsonl --mode visual \
    --limit 50 --out results/probe_visual.jsonl
python scripts/eval_probe.py --data data/match_cad2tgt.jsonl \
    --text results/probe_text.jsonl --visual results/probe_visual.jsonl
