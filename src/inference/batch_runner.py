"""
Batch inference runner: processes patient cohort through Ollama.

Supports two modes:
1. Single-turn (baseline): full timeline stuffed into one prompt per hour
2. Multi-turn (trajectory-aware): hour-by-hour conversation with memory,
   matching the SFT training format
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

from src.prompts.templates import (
    build_incremental_prompts, build_multiturn_messages,
    SYSTEM_PROMPT,
)
from src.sft.prepare_sft_data import TRAJECTORY_SYSTEM_PROMPT
from src.inference.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def run_patient_single_turn(client, patient_hours, ordering="chronological"):
    """
    Baseline mode: each hour is an independent prompt with full history.
    No conversational memory — the model sees the whole timeline each time.
    """
    prompts = build_incremental_prompts(patient_hours, ordering=ordering)
    results = []

    for hour, sys_prompt, user_prompt in prompts:
        result = client.generate(sys_prompt, user_prompt)
        result["hour"] = hour
        result["ordering"] = ordering
        result["mode"] = "single_turn"
        results.append(result)

    return results


def run_patient_multiturn(client, patient_hours, ordering="chronological",
                          system_prompt=None):
    """
    Trajectory-aware mode: multi-turn conversation where each hour is a
    new user turn and the model's prior responses are kept in context.

    This is how the fine-tuned model was trained — it learns to revise
    beliefs across a conversation, not just assess at a single point.
    """
    if system_prompt is None:
        system_prompt = TRAJECTORY_SYSTEM_PROMPT

    turns = build_multiturn_messages(patient_hours, ordering=ordering)
    messages = [{"role": "system", "content": system_prompt}]
    results = []

    for hour, user_msg in turns:
        messages.append({"role": "user", "content": user_msg})

        result = client.chat_multiturn(messages)
        result["hour"] = hour
        result["ordering"] = ordering
        result["mode"] = "multiturn"
        results.append(result)

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": result["response"]})

    return results


def run_batch(config_path="config/paths.yaml", ordering="chronological",
              model_override=None, output_suffix="", mode="multiturn"):
    """
    Run inference on the full sample cohort.

    Args:
        config_path: Path to config YAML.
        ordering: "chronological", "reverse", or "shuffled"
        model_override: Override the model name (e.g., for fine-tuned model).
        output_suffix: Suffix for output filename.
        mode: "single_turn" (baseline) or "multiturn" (trajectory-aware).
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
    logger.info(f"Running {ordering} ({mode}) inference on {sample_size} patients")

    # Init client
    model = model_override or cfg["ollama"]["model_base"]
    client = OllamaClient(host=cfg["ollama"]["host"], model=model)

    if not client.is_available():
        raise RuntimeError(f"Ollama not available at {cfg['ollama']['host']}")

    run_fn = run_patient_multiturn if mode == "multiturn" else run_patient_single_turn

    # Pre-compute sepsis lookup
    sepsis_lookup = dict(zip(sepsis["ICUSTAY_ID"], sepsis["SEPSIS_ONSET_HOUR"]))

    # Run inference
    all_results = []
    for icustay_id in tqdm(sample_ids, desc=f"Inference ({ordering}, {mode})"):
        patient_data = hourly[hourly["ICUSTAY_ID"] == icustay_id]
        if len(patient_data) < 3:
            continue

        patient_results = run_fn(client, patient_data, ordering)
        has_sepsis = icustay_id in sepsis_lookup
        onset_hour = int(sepsis_lookup[icustay_id]) if has_sepsis else None

        for r in patient_results:
            r["icustay_id"] = int(icustay_id)
            r["has_sepsis"] = has_sepsis
            r["sepsis_onset_hour"] = onset_hour

        all_results.extend(patient_results)

    # Save results
    model_tag = model.replace(":", "_")
    fname = f"predictions_{ordering}_{mode}_{model_tag}{output_suffix}.jsonl"
    outpath = results_dir / fname
    with open(outpath, "w") as f:
        for r in all_results:
            row = {k: v for k, v in r.items() if k != "full_response"}
            f.write(json.dumps(row) + "\n")

    logger.info(f"Saved {len(all_results)} predictions to {outpath}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordering", default="chronological",
                        choices=["chronological", "reverse", "shuffled"])
    parser.add_argument("--mode", default="multiturn",
                        choices=["single_turn", "multiturn"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default="config/paths.yaml")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    run_batch(args.config, args.ordering, args.model, args.suffix, args.mode)
