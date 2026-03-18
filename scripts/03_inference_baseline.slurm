#!/bin/bash
#SBATCH --job-name=mimic-infer-base
#SBATCH --output=logs/03_infer_base_%j.out
#SBATCH --error=logs/03_infer_base_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Phase 3: Baseline inference via Ollama (all 3 orderings)
# Requires Ollama running on the same node with llama3.1:8b pulled

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
sleep 15  # wait for server startup

# Ensure model is available
apptainer exec --nv ~/ollama.sif ollama pull llama3.1:8b

# Run all three orderings in both modes (single-turn baseline + multi-turn)
for MODE in single_turn multiturn; do
  for ORDERING in chronological reverse shuffled; do
    echo "=== Running ${ORDERING} (${MODE}) ==="
    python -m src.inference.batch_runner --ordering $ORDERING --mode $MODE
  done
done

# Cleanup
kill $OLLAMA_PID
