#!/bin/bash
#SBATCH --job-name=mimic-sft-prep
#SBATCH --output=logs/02_sft_prep_%j.out
#SBATCH --error=logs/02_sft_prep_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq

# Phase 2: Prepare SFT dataset (CPU-only)

module load python3/3.11.6

cd /SEAS/home/g32999482/emse-6575/mimic-anchoring
source venv/bin/activate

python -m src.sft.prepare_sft_data
