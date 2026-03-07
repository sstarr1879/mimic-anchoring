#!/bin/bash
#SBATCH --job-name=mimic-extract
#SBATCH --output=logs/01_extract_%j.out
#SBATCH --error=logs/01_extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=general

# Phase 1: Cohort extraction (CPU-only, needs RAM for large CSVs)

module load python/3.11  # adjust to your HPC module name

cd /SEAS/home/g32999482/mimic-anchoring
source venv/bin/activate

python -m src.data.extract_cohort
