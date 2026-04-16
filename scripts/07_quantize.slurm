#!/bin/bash
#SBATCH --job-name=mimic-quantize
#SBATCH --output=logs/07_quantize_%j.out
#SBATCH --error=logs/07_quantize_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=superChip
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G

# Merge LoRA adapter and export to quantized GGUF for Ollama deployment.

module load cuda/12.4

cd /SEAS/home/g32999482/emse-6575/mimic-anchoring
source ~/miniforge/bin/activate lora

python scripts/quantize_unsloth.py
