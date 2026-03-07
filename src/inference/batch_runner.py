"""
Batch inference runner: processes patient cohort through Ollama.

Runs incremental assessments for each patient and collects results
for anchoring bias analysis.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

from src.prompts.templates import build_incremental_prompts, build_timeline_prompt, SYSTEM_PROMPT, INCREMENTAL_USER_PROMPT
from src.inference.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def run_patient_incremental(client, patient_hours, ordering="chronological"):
    """
    Run incremental assessment for a single patient.

    Returns list of dicts: [{hour, risk, reasoning, response}, ...]
    """
    prompts = build_incremental_prompts(patient_hours, ordering=ordering)
    results = []

    for hour, sys_prompt, user_prompt in prompts:
        result = client.generate(sys_prompt, user_prompt)
        result["hour"] = hour
        result["ordering"] = ordering
        results.append(result)

    return results


def run_batch(config_path="config/paths.yaml", ordering="chronological",
              model_override=None, output_suffix=""):
    """
    Run inference on the full sample cohort.

    Args:
        config_path: Path to config YAML.
        ordering: "chronological", "reverse", or "shuffled"
        model_override: Override the model name (e.g., for fine-tuned model).
        output_suffix: Suffix for output filename.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    hourly = pd.read_parquet(processed_dir / "hourly_timelines.parquet")
    sepsis = pd.read_parquet(processed_dir / "sepsis_onset.parquet")

    # Sample patients
    all_patients = hourly["ICUSTAY_ID"].unique()
    rng = np.random.RandomState(cfg["anchoring"]["seed"])
    sample_size = min(cfg["anchoring"]["sample_size"], len(all_patients))
    sample_ids = rng.choice(all_patients, size=sample_size, replace=False)
    logger.info(f"Running {ordering} inference on {sample_size} patients")

    # Init client
    model = model_override or cfg["ollama"]["model_base"]
    client = OllamaClient(host=cfg["ollama"]["host"], model=model)

    if not client.is_available():
        raise RuntimeError(f"Ollama not available at {cfg['ollama']['host']}")

    # Run inference
    all_results = []
    for icustay_id in tqdm(sample_ids, desc=f"Inference ({ordering})"):
        patient_data = hourly[hourly["ICUSTAY_ID"] == icustay_id]
        if len(patient_data) < 3:  # skip very short stays
            continue

        patient_results = run_patient_incremental(client, patient_data, ordering)
        for r in patient_results:
            r["icustay_id"] = int(icustay_id)
            has_sepsis = icustay_id in sepsis["ICUSTAY_ID"].values
            r["has_sepsis"] = has_sepsis
            if has_sepsis:
                r["sepsis_onset_hour"] = int(
                    sepsis[sepsis["ICUSTAY_ID"] == icustay_id]["SEPSIS_ONSET_HOUR"].iloc[0]
                )
            else:
                r["sepsis_onset_hour"] = None

        all_results.extend(patient_results)

    # Save results
    fname = f"predictions_{ordering}_{model.replace(':', '_')}{output_suffix}.jsonl"
    outpath = results_dir / fname
    with open(outpath, "w") as f:
        for r in all_results:
            # Remove non-serializable fields
            row = {k: v for k, v in r.items() if k != "full_response"}
            f.write(json.dumps(row) + "\n")

    logger.info(f"Saved {len(all_results)} predictions to {outpath}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordering", default="chronological",
                        choices=["chronological", "reverse", "shuffled"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default="config/paths.yaml")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    run_batch(args.config, args.ordering, args.model, args.suffix)
