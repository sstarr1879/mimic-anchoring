#!/bin/bash
#SBATCH --job-name=mimic-shuf
#SBATCH --output=logs/03_infer_shuf_%j.out
#SBATCH --error=logs/03_infer_shuf_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu

# Shuffled ordering inference (single-turn + multi-turn)

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
sleep 15

# Ensure model is available
apptainer exec --nv ~/ollama.sif ollama pull llama3.1:8b

# Run shuffled ordering in both modes
for MODE in single_turn multiturn; do
  echo "=== Running shuffled (${MODE}) ==="
  python -m src.inference.batch_runner --ordering shuffled --mode $MODE
done

# Cleanup
kill $OLLAMA_PID
