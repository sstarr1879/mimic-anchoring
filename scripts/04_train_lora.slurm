#!/bin/bash
#SBATCH --job-name=mimic-lora
#SBATCH --output=logs/04_lora_%j.out
#SBATCH --error=logs/04_lora_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Phase 3b: LoRA SFT training
# Requires GPU with >= 24GB VRAM (A100, A6000, etc.)

module load python3/3.11.6
module load cuda/12.1

cd /SEAS/home/g32999482/mimic-anchoring
source venv/bin/activate

python -m src.sft.train_lora
