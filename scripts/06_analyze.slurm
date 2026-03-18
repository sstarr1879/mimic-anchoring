#!/bin/bash
#SBATCH --job-name=mimic-analyze
#SBATCH --output=logs/06_analyze_%j.out
#SBATCH --error=logs/06_analyze_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq

# Phase 5: Run anchoring bias analysis on all collected predictions

module load python3/3.11.6

cd /SEAS/home/g32999482/emse-6575/mimic-anchoring
source venv/bin/activate

python -m src.bias.anchoring_metrics
