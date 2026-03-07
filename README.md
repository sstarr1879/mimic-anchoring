# Measuring and Mitigating First Impression Bias in ICU Patient Monitoring

![python](https://img.shields.io/badge/python-3.11+-green)
![status](https://img.shields.io/badge/status-experimental-orange)

This project investigates **anchoring bias** (first impression bias) in large language models used for ICU sepsis prediction. When a patient appears stable at admission, both human clinicians and AI systems tend to lock onto "this patient is fine" — even as later vital signs suggest developing sepsis. We measure this effect in LLaMA 3.1 8B and test interventions to mitigate it.

## Three-Part Measurement Framework

1. **Counterfactual Ordering** — Present the same patient's data in chronological, reverse, and shuffled order. If early information dominates the final prediction despite identical evidence, the model is anchored.

2. **Belief Update Elasticity** — Feed vitals incrementally (hour-by-hour) and measure how much the model's risk estimate shifts when contradictory evidence arrives. Anchored models barely budge; flexible models adjust proportionally.

3. **Explanation Drift** — Compare the semantic content of the model's reasoning at early vs. late timesteps. Genuine revision changes the narrative; anchored systems retrofit justifications to match prior conclusions.

## Interventions

- **Forced context reset** — Truncate early history every N hours so the model can't anchor on stale impressions
- **Re-prompting** — Explicitly instruct the model to reassess from scratch at regular intervals
- **LoRA SFT comparison** — Fine-tune on sepsis assessment data and measure whether training amplifies or reduces anchoring

## Data

[MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/) — 53,423 hospital admissions with timestamped vital signs, lab results, medications, and clinical notes. Sepsis cohort extracted using Sepsis-3 criteria (Singer et al., JAMA 2016): suspected infection + SOFA >= 2, yielding ~5,000-7,000 cases.

**Access requires PhysioNet credentialing** (CITI training + data use agreement).

## Project Structure

```
mimic-anchoring/
├── config/
│   └── paths.yaml                  # All HPC paths and hyperparameters
├── data/
│   ├── raw/                        # MIMIC-III .csv.gz files
│   ├── processed/                  # Extracted cohort, timelines, SFT data
│   ├── models/                     # LoRA adapters and GGUF exports
│   └── results/                    # Predictions and analysis outputs
├── src/
│   ├── data/
│   │   └── extract_cohort.py       # Sepsis-3 cohort extraction from MIMIC CSVs
│   ├── prompts/
│   │   └── templates.py            # Prompt formatting for patient timelines
│   ├── inference/
│   │   ├── ollama_client.py        # Ollama API wrapper + response parsing
│   │   └── batch_runner.py         # Batch inference across patient cohort
│   ├── bias/
│   │   └── anchoring_metrics.py    # Ordering effect, elasticity, explanation drift
│   ├── sft/
│   │   ├── prepare_sft_data.py     # Generate SFT training data from SOFA ground truth
│   │   ├── train_lora.py           # LoRA fine-tuning via Unsloth
│   │   └── deploy_ollama.py        # GGUF → Ollama model registration
│   └── interventions/
│       └── context_reset.py        # Context window reset + re-prompting strategies
├── scripts/
│   ├── setup_hpc.sh                # One-time HPC environment setup
│   ├── 01_extract_cohort.slurm     # Phase 1: cohort extraction
│   ├── 02_prepare_sft.slurm        # Phase 2: SFT dataset construction
│   ├── 03_inference_baseline.slurm # Phase 3: zero-shot anchoring measurement
│   ├── 04_train_lora.slurm         # Phase 4: LoRA fine-tuning
│   ├── 05_deploy_and_infer_finetuned.slurm  # Phase 5: post-SFT measurement
│   └── 06_analyze.slurm            # Phase 6: anchoring bias analysis
├── notebooks/                      # EDA and visualization (TBD)
└── requirements.txt
```

## Setup

### Prerequisites
- HPC access with SLURM scheduler and GPU nodes (>= 24GB VRAM for training)
- Python 3.11+
- [Ollama](https://ollama.com/) installed on GPU nodes
- MIMIC-III data access via PhysioNet

### Installation

```bash
# Clone to HPC
cd /SEAS/home/g32999482/
git clone <repo-url> mimic-anchoring
cd mimic-anchoring

# Run setup script
bash scripts/setup_hpc.sh

# Place MIMIC-III .csv.gz files in data/raw/
```

### Execution

Run the SLURM jobs in order — each depends on the output of the previous step:

```bash
sbatch scripts/01_extract_cohort.slurm         # ~2-4 hours, CPU
sbatch scripts/02_prepare_sft.slurm            # ~30 min, CPU
sbatch scripts/03_inference_baseline.slurm      # ~12-24 hours, 1 GPU
sbatch scripts/04_train_lora.slurm              # ~4-8 hours, 1 GPU
sbatch scripts/05_deploy_and_infer_finetuned.slurm  # ~12-24 hours, 1 GPU
sbatch scripts/06_analyze.slurm                 # ~1 hour, CPU
```

Check job status with `squeue -u $USER` and logs in `logs/`.

## Model

- **Base**: LLaMA 3.1 8B Instruct via Ollama
- **Fine-tuned**: LoRA (r=16, alpha=32) trained with Unsloth, exported to GGUF Q4_K_M for Ollama serving

## Key References

- Singer et al., "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)," JAMA 2016
- Nemati et al., "An Interpretable Machine Learning Model for Accurate Prediction of Sepsis in the ICU," Critical Care Medicine 2018
- Komorowski et al., "The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care," Nature Medicine 2018
