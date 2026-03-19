#!/bin/bash
#SBATCH --job-name=mimic-ft-infer
#SBATCH --output=logs/05_infer_ft_%j.out
#SBATCH --error=logs/05_infer_ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclusive

# Run a single ordering for fine-tuned model (multi-turn only, matches SFT training format)
# Usage: sbatch --export=ORDERING=chronological scripts/05_infer_finetuned_single.slurm

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

# Deploy fine-tuned model
python -m src.sft.deploy_ollama

FT_MODEL="llama3.1-sepsis:8b"

echo "=== Fine-tuned: ${ORDERING} (multiturn) ==="
python -m src.inference.batch_runner --ordering $ORDERING --mode multiturn --model "$FT_MODEL" --suffix "_finetuned"

kill $OLLAMA_PID
