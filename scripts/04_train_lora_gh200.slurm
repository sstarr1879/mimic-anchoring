#!/bin/bash
#SBATCH --job-name=mimic-lora
#SBATCH --output=logs/04_lora_%j.out
#SBATCH --error=logs/04_lora_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=superChip
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --exclusive

# Phase 3b: LoRA SFT training on GH200 (Grace Hopper) nodes

module load cuda/12.8

cd /SEAS/home/g32999482/emse-6575/mimic-anchoring
source ~/miniforge/bin/activate lora

# Clear Unsloth compiled cache to pick up patches
rm -rf unsloth_compiled_cache/

python -m src.sft.train_lora
