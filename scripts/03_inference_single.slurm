#!/bin/bash
#SBATCH --job-name=mimic-infer
#SBATCH --output=logs/03_infer_%j.out
#SBATCH --error=logs/03_infer_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclusive

# Run a single ordering+mode combo
# Usage: sbatch --export=ORDERING=chronological,MODE=single_turn scripts/03_inference_single.slurm

module load python3/3.11.6
module load cuda/12.2
module load apptainer

cd /SEAS/home/g32999482/emse-6575/mimic-anchoring
source venv/bin/activate

# Use unique port per job to avoid conflicts
OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 1000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_MODELS=$HOME/ollama_models
mkdir -p $OLLAMA_MODELS

# Start Ollama server via apptainer in background
apptainer exec --nv ~/ollama.sif ollama serve &
OLLAMA_PID=$!
sleep 15

# Ensure model is available
apptainer exec --nv ~/ollama.sif ollama pull llama3.1:8b

echo "=== Running ${ORDERING} (${MODE}) on port ${OLLAMA_PORT} ==="
python -m src.inference.batch_runner --ordering $ORDERING --mode $MODE --ollama-host "http://127.0.0.1:${OLLAMA_PORT}"

kill $OLLAMA_PID
