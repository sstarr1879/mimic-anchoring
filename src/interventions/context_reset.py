"""
Phase 5: Intervention strategies to mitigate anchoring bias.

Implements:
1. Forced context reset — truncate early history every N hours
2. Re-prompting — explicit instruction to reassess from scratch
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm

from src.prompts.templates import (
    SYSTEM_PROMPT, REASSESS_PROMPT,
    build_timeline_prompt, build_incremental_prompts,
)
from src.inference.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def run_with_context_reset(client, patient_hours, reset_interval=6):
    """
    Run incremental assessment with periodic context resets.

    Every `reset_interval` hours, the model only sees the last
    `reset_interval` hours of data instead of the full history.
    """
    df = patient_hours.sort_values("HOUR")
    hours = sorted(df["HOUR"].unique())
    results = []

    for h in hours:
        # Determine context window
        if reset_interval is not None:
            window_start = max(0, h - reset_interval)
            context_hours = df[(df["HOUR"] >= window_start) & (df["HOUR"] <= h)]
        else:
            context_hours = df[df["HOUR"] <= h]

        timeline = build_timeline_prompt(context_hours, up_to_hour=h)
        user_prompt = f"Based on the patient data so far, assess the current sepsis risk.\n\n{timeline}\n\nProvide your assessment for the most recent hour."

        result = client.generate(SYSTEM_PROMPT, user_prompt)
        result["hour"] = h
        result["context_window_start"] = window_start if reset_interval else 0
        result["intervention"] = f"reset_{reset_interval}h"
        results.append(result)

    return results


def run_with_reassess_prompt(client, patient_hours, reassess_interval=6):
    """
    Run incremental assessment with periodic re-prompting.

    Every `reassess_interval` hours, instead of the normal prompt,
    use the REASSESS_PROMPT that explicitly asks the model to
    ignore prior context and evaluate fresh.
    """
    df = patient_hours.sort_values("HOUR")
    hours = sorted(df["HOUR"].unique())
    results = []

    for h in hours:
        timeline = build_timeline_prompt(patient_hours, up_to_hour=h)

        if h > 0 and h % reassess_interval == 0:
            user_prompt = REASSESS_PROMPT.format(timeline=timeline)
            intervention_type = "reassess"
        else:
            user_prompt = f"Based on the patient data so far, assess the current sepsis risk.\n\n{timeline}\n\nProvide your assessment for the most recent hour."
            intervention_type = "normal"

        result = client.generate(SYSTEM_PROMPT, user_prompt)
        result["hour"] = h
        result["intervention"] = intervention_type
        results.append(result)

    return results


def run_interventions_batch(config_path="config/paths.yaml", model_override=None):
    """Run all intervention strategies on the sample cohort."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_parquet(processed_dir / "hourly_timelines.parquet")
    sepsis = pd.read_parquet(processed_dir / "sepsis_onset.parquet")

    all_patients = hourly["ICUSTAY_ID"].unique()
    rng = np.random.RandomState(cfg["anchoring"]["seed"])
    sample_size = min(cfg["anchoring"]["sample_size"], len(all_patients))
    sample_ids = rng.choice(all_patients, size=sample_size, replace=False)

    model = model_override or cfg["ollama"]["model_base"]
    client = OllamaClient(host=cfg["ollama"]["host"], model=model)
    reset_hours = cfg["anchoring"]["reset_interval_hours"]

    for intervention_name, run_fn, kwargs in [
        (f"context_reset_{reset_hours}h", run_with_context_reset, {"reset_interval": reset_hours}),
        (f"reassess_{reset_hours}h", run_with_reassess_prompt, {"reassess_interval": reset_hours}),
    ]:
        logger.info(f"Running intervention: {intervention_name}")
        all_results = []

        for icustay_id in tqdm(sample_ids, desc=intervention_name):
            patient_data = hourly[hourly["ICUSTAY_ID"] == icustay_id]
            if len(patient_data) < 3:
                continue

            patient_results = run_fn(client, patient_data, **kwargs)
            for r in patient_results:
                r["icustay_id"] = int(icustay_id)
                has_sepsis = icustay_id in sepsis["ICUSTAY_ID"].values
                r["has_sepsis"] = has_sepsis

            all_results.extend(patient_results)

        model_tag = model.replace(":", "_")
        outpath = results_dir / f"predictions_{intervention_name}_{model_tag}.jsonl"
        with open(outpath, "w") as f:
            for r in all_results:
                row = {k: v for k, v in r.items() if k != "full_response"}
                f.write(json.dumps(row) + "\n")
        logger.info(f"Saved {len(all_results)} results to {outpath}")


if __name__ == "__main__":
    run_interventions_batch()
