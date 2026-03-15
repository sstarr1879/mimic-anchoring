#!/bin/bash
# Run this once on the HPC to set up the environment

set -e

echo "=== Setting up mimic-anchoring environment ==="

# Create directories
mkdir -p logs

# Create virtual environment
module load python3/3.11.6
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Ollama (if not system-installed)
# Uncomment if needed:
# curl -fsSL https://ollama.com/install.sh | sh

# Pull base model
# ollama pull llama3.1:8b

# Create data subdirectories
mkdir -p /SEAS/home/g32999482/mimic-anchoring/data/{raw,processed,models,results}
echo ""
echo "=== Setup complete ==="
echo "NEXT STEPS:"
echo "1. Edit config/paths.yaml with your HPC paths"
echo "2. Ensure MIMIC-III CSVs are at the path specified in mimic_raw_dir"
echo "3. Run: sbatch scripts/01_extract_cohort.slurm"
