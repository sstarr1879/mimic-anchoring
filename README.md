# Measuring and Mitigating First Impression Bias in ICU Patient Monitoring

![python](https://img.shields.io/badge/python-3.11+-green)
![status](https://img.shields.io/badge/status-experimental-orange)

This project investigates **anchoring bias** (first impression bias) in large language models used for ICU sepsis prediction. When a patient appears stable at admission, both human clinicians and AI systems tend to lock onto "this patient is fine" — even as later vital signs suggest developing sepsis. We measure this effect in LLaMA 3.1 8B and test whether **trajectory-aware post-training** (LoRA SFT on expert revision traces) can teach the model to revise its beliefs correctly.

## Approach

### Three-Part Measurement Framework

1. **Counterfactual Ordering** — Present the same patient's data in chronological, reverse, and shuffled order. If early information dominates the final prediction despite identical evidence, the model is anchored.

2. **Belief Update Elasticity** — Feed vitals hour-by-hour in a multi-turn conversation and measure how much the model's risk estimate shifts when contradictory evidence arrives. Anchored models barely budge; flexible models adjust proportionally.

3. **Explanation Drift** — Compare the semantic content of the model's reasoning at early vs. late timesteps. Genuine revision changes the narrative; anchored systems retrofit justifications to match prior conclusions.

### Trajectory-Aware SFT

Standard SFT trains models to assess risk at isolated snapshots. Our approach instead trains on **full multi-turn expert traces** — complete ICU stay conversations where an ideal assessor demonstrates how to revise beliefs as new evidence arrives:

```
system  → You are an ICU monitor... reference your previous assessment...
user    → Hour 0:  HR=82, MAP=78, Lactate=1.1
assistant → RISK: 0.15 | PREVIOUS: N/A    | DELTA: +0.00 | REASONING: Stable...
user    → Hour 4:  HR=95, MAP=72, Lactate=1.8
assistant → RISK: 0.32 | PREVIOUS: 0.15   | DELTA: +0.17 | REASONING: Revising upward,
            HR trending up and lactate rising...
user    → Hour 8:  HR=112, MAP=63, Lactate=4.2
assistant → RISK: 0.72 | PREVIOUS: 0.32   | DELTA: +0.40 | REASONING: Significant
            revision — prior assessment underestimated risk. Lactate doubled,
            now hypotensive...
```

The model explicitly learns to reference prior estimates (`PREVIOUS`), quantify changes (`DELTA`), and explain **why** it is revising — including acknowledging when earlier assessments were wrong.

### Interventions

- **Forced context reset** — Truncate early history every N hours so the model can't anchor on stale impressions
- **Re-prompting** — Explicitly instruct the model to reassess from scratch at regular intervals

### Experimental Design

Baseline (zero-shot) and fine-tuned models are each evaluated across two inference modes and three orderings, yielding a comparison matrix:

| | Chronological | Reverse | Shuffled |
|---|---|---|---|
| **Base zero-shot (single-turn)** | anchoring baseline | counterfactual | noise test |
| **Base zero-shot (multi-turn)** | conversational baseline | counterfactual | noise test |
| **SFT trajectory-aware (multi-turn)** | does SFT reduce anchoring? | revision under reversal | robustness |
| **Context reset (every 6h)** | truncation vs anchoring | truncation vs anchoring | — |
| **Re-prompting (every 6h)** | explicit reassessment | explicit reassessment | — |

The **context reset** intervention truncates conversation history every N hours, preventing the model from anchoring on stale early impressions. **Re-prompting** keeps full history but explicitly instructs the model to reassess from scratch at regular intervals. Both are applied to the fine-tuned model in step 5 and compared against the unmodified SFT results to determine whether architectural interventions provide additional debiasing beyond training alone.

## Data

[MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/) — 53,423 hospital admissions with timestamped vital signs, lab results, medications, and clinical notes. Sepsis cohort extracted using Sepsis-3 criteria (Singer et al., JAMA 2016): suspected infection + SOFA >= 2, yielding ~5,000-7,000 cases.

**Access requires PhysioNet credentialing** (CITI training + data use agreement).

### Key MIMIC-III Tables Used

| Table | Purpose |
|---|---|
| `CHARTEVENTS` | Vitals (HR, MAP, temp, SpO2, RR) |
| `LABEVENTS` | Labs (lactate, WBC, creatinine, platelets, bilirubin) |
| `ICUSTAYS` | ICU stay metadata and timestamps |
| `ADMISSIONS` / `PATIENTS` | Demographics |
| `DIAGNOSES_ICD` | Diagnosis codes |
| `MICROBIOLOGYEVENTS` | Suspected infection (Sepsis-3 criteria) |
| `PRESCRIPTIONS` | Antibiotic orders (Sepsis-3 criteria) |

## Project Structure

```
mimic-anchoring/
├── config/
│   └── paths.yaml                  # All HPC paths and hyperparameters
├── data/
│   ├── raw/                        # MIMIC-III .csv.gz files
│   ├── processed/                  # Extracted cohort, timelines, SFT traces
│   ├── models/                     # LoRA adapters and GGUF exports
│   └── results/                    # Predictions and analysis outputs
├── src/
│   ├── data/
│   │   └── extract_cohort.py       # Sepsis-3 cohort extraction from MIMIC CSVs
│   ├── prompts/
│   │   └── templates.py            # Prompt formatting (single-turn + multi-turn)
│   ├── inference/
│   │   ├── ollama_client.py        # Ollama API wrapper (generate + chat_multiturn)
│   │   └── batch_runner.py         # Batch inference (single-turn and multi-turn modes)
│   ├── bias/
│   │   └── anchoring_metrics.py    # Ordering effect, elasticity, explanation drift
│   ├── sft/
│   │   ├── prepare_sft_data.py     # Generate multi-turn expert traces for SFT
│   │   ├── train_lora.py           # LoRA fine-tuning via Unsloth
│   │   └── deploy_ollama.py        # GGUF → Ollama model registration
│   └── interventions/
│       └── context_reset.py        # Context window reset + re-prompting strategies
├── scripts/
│   ├── setup_hpc.sh                # One-time HPC environment setup
│   ├── 01_extract_cohort.slurm     # Cohort extraction (CPU)
│   ├── 02_prepare_sft.slurm        # SFT trace construction (CPU)
│   ├── 03_inference_baseline.slurm # Zero-shot: single-turn + multi-turn x 3 orderings (GPU)
│   ├── 04_train_lora.slurm         # LoRA fine-tuning on expert traces (GPU)
│   ├── 05_deploy_and_infer_finetuned.slurm  # Post-SFT multi-turn x 3 orderings (GPU)
│   └── 06_analyze.slurm            # Anchoring bias analysis (CPU)
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

# Run setup script (creates venv, installs deps, makes data dirs)
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

- **Base**: LLaMA 3.1 8B Instruct via Ollama (zero-shot)
- **Fine-tuned**: LoRA (r=16, alpha=32) trained on trajectory-aware expert traces via Unsloth, exported to GGUF Q4_K_M for Ollama serving

## Key References

- Singer et al., "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)," JAMA 2016
- Nemati et al., "An Interpretable Machine Learning Model for Accurate Prediction of Sepsis in the ICU," Critical Care Medicine 2018
- Komorowski et al., "The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care," Nature Medicine 2018
