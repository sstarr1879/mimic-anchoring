#!/bin/bash
#SBATCH --job-name=mimic-infer-ft
#SBATCH --output=logs/05_infer_ft_%j.out
#SBATCH --error=logs/05_infer_ft_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Phase 4: Deploy fine-tuned model to Ollama and run anchoring experiments

module load python/3.11
module load cuda/12.1

cd /SEAS/home/g32999482/mimic-anchoring
source venv/bin/activate

# Start Ollama
ollama serve &
OLLAMA_PID=$!
sleep 10

# Deploy fine-tuned model
python -m src.sft.deploy_ollama

FT_MODEL="llama3.1-sepsis:8b"

# Run orderings on fine-tuned model
echo "=== Fine-tuned: chronological ==="
python -m src.inference.batch_runner --ordering chronological --model "$FT_MODEL" --suffix "_finetuned"

echo "=== Fine-tuned: reverse ==="
python -m src.inference.batch_runner --ordering reverse --model "$FT_MODEL" --suffix "_finetuned"

echo "=== Fine-tuned: shuffled ==="
python -m src.inference.batch_runner --ordering shuffled --model "$FT_MODEL" --suffix "_finetuned"

# Run interventions
echo "=== Interventions ==="
python -m src.interventions.context_reset --model "$FT_MODEL"

kill $OLLAMA_PID
