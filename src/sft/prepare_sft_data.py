"""
Phase 3a: Prepare SFT training data for LoRA fine-tuning.

Converts patient timelines + SOFA-based ground truth into
instruction-tuning format for LLaMA 3.1 8B.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import yaml

from src.prompts.templates import (
    SYSTEM_PROMPT, build_timeline_prompt, FEATURE_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)


def sofa_to_risk_label(sofa_score, max_sofa=15):
    """Convert SOFA score to a normalized risk probability."""
    # Sigmoid-like mapping centered around SOFA=2 (sepsis threshold)
    return 1.0 / (1.0 + np.exp(-0.5 * (sofa_score - 2)))


def generate_target_response(risk, hour, patient_hours, sofa_at_hour):
    """
    Generate a target response for SFT training.

    Uses clinical heuristics to create reasonable reasoning text
    based on the actual vital signs and SOFA score.
    """
    row = patient_hours[patient_hours["HOUR"] == hour]
    if len(row) == 0:
        return f"RISK: {risk:.2f}\nREASONING: Insufficient data for detailed assessment."

    row = row.iloc[0]

    # Build reasoning from available features
    observations = []
    if "HR" in row.index and pd.notna(row.get("HR")):
        hr = row["HR"]
        if hr > 100:
            observations.append(f"tachycardia (HR {hr:.0f})")
        elif hr < 60:
            observations.append(f"bradycardia (HR {hr:.0f})")
        else:
            observations.append(f"normal heart rate (HR {hr:.0f})")

    if "MAP" in row.index and pd.notna(row.get("MAP")):
        m = row["MAP"]
        if m < 65:
            observations.append(f"hypotension (MAP {m:.0f})")
        elif m < 70:
            observations.append(f"borderline low MAP ({m:.0f})")
        else:
            observations.append(f"adequate perfusion (MAP {m:.0f})")

    if "LACTATE" in row.index and pd.notna(row.get("LACTATE")):
        lac = row["LACTATE"]
        if lac > 4:
            observations.append(f"severely elevated lactate ({lac:.1f})")
        elif lac > 2:
            observations.append(f"elevated lactate ({lac:.1f})")
        else:
            observations.append(f"normal lactate ({lac:.1f})")

    if "WBC" in row.index and pd.notna(row.get("WBC")):
        wbc = row["WBC"]
        if wbc > 12:
            observations.append(f"leukocytosis (WBC {wbc:.1f})")
        elif wbc < 4:
            observations.append(f"leukopenia (WBC {wbc:.1f})")

    if "TEMP_C" in row.index and pd.notna(row.get("TEMP_C")):
        t = row["TEMP_C"]
        if t > 38.3:
            observations.append(f"fever ({t:.1f}C)")
        elif t < 36:
            observations.append(f"hypothermia ({t:.1f}C)")

    if "CREATININE" in row.index and pd.notna(row.get("CREATININE")):
        cr = row["CREATININE"]
        if cr > 2:
            observations.append(f"renal dysfunction (Cr {cr:.1f})")

    if observations:
        reasoning = f"Assessment based on: {', '.join(observations)}. "
    else:
        reasoning = "Limited observations available. "

    if risk > 0.7:
        reasoning += "Multiple indicators suggest high sepsis risk requiring immediate attention."
    elif risk > 0.4:
        reasoning += "Some concerning trends warrant close monitoring for sepsis development."
    else:
        reasoning += "Current indicators suggest relatively low sepsis risk."

    return f"RISK: {risk:.2f}\nREASONING: {reasoning}"


def prepare_sft_dataset(config_path="config/paths.yaml"):
    """
    Create SFT training dataset from processed MIMIC-III data.

    Format: JSONL with {messages: [{role, content}, ...]} per example.
    Each example is a patient timeline up to hour H with the target
    assessment at hour H.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])
    out_dir = Path(cfg["processed_dir"]) / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_parquet(processed_dir / "hourly_timelines.parquet")
    sofa = pd.read_parquet(processed_dir / "sofa_scores.parquet")
    sepsis = pd.read_parquet(processed_dir / "sepsis_onset.parquet")

    # Merge SOFA scores into hourly
    hourly_sofa = hourly.merge(
        sofa[["ICUSTAY_ID", "HOUR", "SOFA"]],
        on=["ICUSTAY_ID", "HOUR"],
        how="left",
    )
    hourly_sofa["SOFA"] = hourly_sofa["SOFA"].fillna(0)

    all_patients = hourly_sofa["ICUSTAY_ID"].unique()
    rng = np.random.RandomState(cfg["anchoring"]["seed"])
    rng.shuffle(all_patients)

    # 80/10/10 split
    n = len(all_patients)
    train_ids = set(all_patients[: int(0.8 * n)])
    val_ids = set(all_patients[int(0.8 * n): int(0.9 * n)])
    test_ids = set(all_patients[int(0.9 * n):])

    logger.info(f"Patients — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        examples = []
        for icustay_id in split_ids:
            patient = hourly_sofa[hourly_sofa["ICUSTAY_ID"] == icustay_id].sort_values("HOUR")
            hours = sorted(patient["HOUR"].unique())

            # Generate training examples at multiple time points
            # (every 4 hours + final hour to keep dataset manageable)
            sample_hours = [h for h in hours if h % 4 == 0 or h == hours[-1]]
            for target_hour in sample_hours:
                sofa_val = patient[patient["HOUR"] == target_hour]["SOFA"].values
                if len(sofa_val) == 0:
                    continue
                sofa_val = sofa_val[0]
                risk = sofa_to_risk_label(sofa_val)

                timeline = build_timeline_prompt(patient, up_to_hour=target_hour)
                user_msg = f"Based on the patient data so far, assess the current sepsis risk.\n\n{timeline}\n\nProvide your assessment for the most recent hour."
                target = generate_target_response(risk, target_hour, patient, sofa_val)

                example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": target},
                    ],
                    "icustay_id": int(icustay_id),
                    "hour": int(target_hour),
                    "sofa": float(sofa_val),
                    "risk": float(risk),
                }
                examples.append(example)

        # Save
        outpath = out_dir / f"{split_name}.jsonl"
        with open(outpath, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        logger.info(f"{split_name}: {len(examples)} examples -> {outpath}")

    logger.info("SFT dataset preparation complete")


if __name__ == "__main__":
    prepare_sft_dataset()
