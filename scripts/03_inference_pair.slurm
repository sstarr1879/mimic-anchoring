#!/bin/bash
#SBATCH --job-name=mimic-infer
#SBATCH --output=logs/03_infer_%j.out
#SBATCH --error=logs/03_infer_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclusive

# Run both modes for a single ordering (one job per GPU node)
# Usage: sbatch --export=ORDERING=chronological scripts/03_inference_pair.slurm

module load python3/3.11.6
module load cuda/12.2
module load apptainer

cd /SEAS/home/g32999482/emse-6575/mimic-anchoring
source venv/bin/activate

# Start Ollama server via apptainer in background
export OLLAMA_MODELS=$HOME/ollama_models
mkdir -p $OLLAMA_MODELS
apptainer exec --nv ~/ollama.sif ollama serve &
OLLAMA_PID=$!
sleep 20

# Ensure model is available
apptainer exec --nv ~/ollama.sif ollama pull llama3.1:8b

echo "=== Running ${ORDERING} (single_turn) ==="
python -m src.inference.batch_runner --ordering $ORDERING --mode single_turn

echo "=== Running ${ORDERING} (multiturn) ==="
python -m src.inference.batch_runner --ordering $ORDERING --mode multiturn

kill $OLLAMA_PID
