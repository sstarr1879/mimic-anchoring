"""
Phase 3a: Prepare trajectory-aware SFT training data for LoRA fine-tuning.

Each training example is a full multi-turn conversation representing an
expert trace through a patient's ICU stay. The model learns not just to
assess risk at a point in time, but to *revise its beliefs correctly*
as new evidence arrives — including explicitly acknowledging when prior
assessments were wrong and why.

Format:
  system → You are an ICU monitor...
  user   → Hour 0 vitals...
  assistant → RISK: 0.15 / REASONING: stable...
  user   → Hour 1 vitals...
  assistant → RISK: 0.18 / REASONING: slight uptick, revising from 0.15...
  ...
  user   → Hour 12 vitals...
  assistant → RISK: 0.72 / REASONING: significant deterioration since hour 8,
              revising sharply upward from 0.45 because lactate doubled...
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import yaml

from src.prompts.templates import (
    SYSTEM_PROMPT, format_hour_observations, FEATURE_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)


TRAJECTORY_SYSTEM_PROMPT = """You are an ICU patient monitoring system performing continuous sepsis surveillance. You receive vital signs and lab values hour-by-hour and must maintain a running sepsis risk assessment.

At each new observation, you must:
1. State your updated sepsis risk probability (0.0 to 1.0)
2. Explicitly reference your previous assessment and explain what changed
3. If the evidence contradicts your prior assessment, acknowledge the revision and explain why

Format your response exactly as:
RISK: <probability>
PREVIOUS: <your last risk estimate, or "N/A" if first assessment>
DELTA: <change from previous, e.g. "+0.15" or "-0.05">
REASONING: <explain what new evidence changed your assessment and why>"""


def sofa_to_risk_label(sofa_score):
    """Convert SOFA score to a normalized risk probability."""
    return 1.0 / (1.0 + np.exp(-0.5 * (sofa_score - 2)))


def describe_trend(current_val, prev_val, feature_name):
    """Describe the trend of a feature between two timepoints."""
    if prev_val is None or np.isnan(prev_val) or np.isnan(current_val):
        return None
    diff = current_val - prev_val
    pct = abs(diff / prev_val) * 100 if prev_val != 0 else 0

    if pct < 5:
        return f"{feature_name} stable"
    elif diff > 0:
        return f"{feature_name} increased ({prev_val:.1f} -> {current_val:.1f})"
    else:
        return f"{feature_name} decreased ({prev_val:.1f} -> {current_val:.1f})"


def generate_expert_response(risk, prev_risk, hour, current_row, prev_row, patient_hours):
    """
    Generate an expert trace response that explicitly revises from the prior assessment.
    This teaches the model HOW to update beliefs, not just what the right answer is.
    """
    delta = risk - prev_risk if prev_risk is not None else 0.0
    prev_str = f"{prev_risk:.2f}" if prev_risk is not None else "N/A"
    delta_str = f"{delta:+.2f}" if prev_risk is not None else "+0.00"

    # Identify what changed since last observation
    trend_notes = []
    feature_cols = [c for c in current_row.index if c in FEATURE_DISPLAY_NAMES]

    if prev_row is not None:
        for feat in feature_cols:
            curr_val = current_row.get(feat)
            prev_val = prev_row.get(feat)
            if pd.notna(curr_val) and pd.notna(prev_val):
                trend = describe_trend(curr_val, prev_val, FEATURE_DISPLAY_NAMES[feat])
                if trend and "stable" not in trend:
                    trend_notes.append(trend)

    # Build clinical reasoning
    observations = []
    if "HR" in current_row.index and pd.notna(current_row.get("HR")):
        hr = current_row["HR"]
        if hr > 100:
            observations.append(f"tachycardia (HR {hr:.0f})")
        elif hr > 90:
            observations.append(f"elevated heart rate (HR {hr:.0f})")
    if "MAP" in current_row.index and pd.notna(current_row.get("MAP")):
        m = current_row["MAP"]
        if m < 65:
            observations.append(f"hypotension (MAP {m:.0f})")
    if "LACTATE" in current_row.index and pd.notna(current_row.get("LACTATE")):
        lac = current_row["LACTATE"]
        if lac > 4:
            observations.append(f"severely elevated lactate ({lac:.1f})")
        elif lac > 2:
            observations.append(f"elevated lactate ({lac:.1f})")
    if "WBC" in current_row.index and pd.notna(current_row.get("WBC")):
        wbc = current_row["WBC"]
        if wbc > 12:
            observations.append(f"leukocytosis (WBC {wbc:.1f})")
        elif wbc < 4:
            observations.append(f"leukopenia (WBC {wbc:.1f})")
    if "TEMP_C" in current_row.index and pd.notna(current_row.get("TEMP_C")):
        t = current_row["TEMP_C"]
        if t > 38.3:
            observations.append(f"fever ({t:.1f}C)")
    if "CREATININE" in current_row.index and pd.notna(current_row.get("CREATININE")):
        cr = current_row["CREATININE"]
        if cr > 2:
            observations.append(f"renal dysfunction (Cr {cr:.1f})")
    # Treatment observations
    norepi_now = current_row.get("TX_NOREPI_RATE", 0) or 0
    if pd.notna(norepi_now) and norepi_now > 0:
        observations.append(f"on norepinephrine ({norepi_now:.2f} mcg/kg/min)")
    n_pressors = current_row.get("TX_VASO_N_AGENTS", 0) or 0
    if pd.notna(n_pressors) and n_pressors >= 2:
        observations.append(f"on {int(n_pressors)} vasopressors")
    if current_row.get("TX_VENT_INVASIVE"):
        observations.append("mechanically ventilated")
    uo = current_row.get("TX_URINE_ML")
    if pd.notna(uo) and uo < 20 and uo >= 0:
        observations.append(f"oliguric (UO {uo:.0f} mL/h)")

    # Detect treatment transitions vs prior hour — these are the key
    # de-anchoring training signals.
    tx_events = []
    if prev_row is not None:
        prev_norepi = prev_row.get("TX_NOREPI_RATE", 0) or 0
        if pd.isna(prev_norepi):
            prev_norepi = 0
        if pd.isna(norepi_now):
            norepi_now = 0
        if prev_norepi == 0 and norepi_now > 0:
            tx_events.append(f"norepinephrine started at {norepi_now:.2f} mcg/kg/min")
        elif prev_norepi > 0 and norepi_now == 0:
            tx_events.append("norepinephrine weaned off")
        elif norepi_now > prev_norepi * 1.5 and prev_norepi > 0:
            tx_events.append(
                f"norepinephrine escalated ({prev_norepi:.2f} -> {norepi_now:.2f})"
            )
        elif norepi_now > 0 and norepi_now < prev_norepi * 0.5:
            tx_events.append(
                f"norepinephrine weaning ({prev_norepi:.2f} -> {norepi_now:.2f})"
            )
        if not prev_row.get("TX_VENT_INVASIVE") and current_row.get("TX_VENT_INVASIVE"):
            tx_events.append("intubated")
        elif prev_row.get("TX_VENT_INVASIVE") and not current_row.get("TX_VENT_INVASIVE"):
            tx_events.append("extubated")

    # Construct reasoning that explicitly addresses belief revision
    reasoning_parts = []

    if prev_risk is None:
        # First assessment
        if observations:
            reasoning_parts.append(f"Initial assessment based on: {', '.join(observations)}.")
        if risk < 0.3:
            reasoning_parts.append("Vitals are within acceptable range. Low initial sepsis risk.")
        elif risk < 0.6:
            reasoning_parts.append("Some mildly concerning values. Moderate vigilance warranted.")
        else:
            reasoning_parts.append("Multiple concerning indicators on admission. Close monitoring required.")
    else:
        # Revision — the key training signal
        abs_delta = abs(delta)
        if abs_delta < 0.05:
            reasoning_parts.append(f"Minimal change from prior assessment of {prev_str}.")
            if observations:
                reasoning_parts.append(f"Current findings: {', '.join(observations)}.")
            reasoning_parts.append("No significant revision warranted.")
        elif delta > 0:
            # Risk increasing — model must learn to revise UPWARD
            reasoning_parts.append(
                f"Revising upward from {prev_str} to {risk:.2f}."
            )
            if tx_events:
                reasoning_parts.append(
                    f"Treatment escalation reflects clinical concern: {'; '.join(tx_events)}."
                )
            if trend_notes:
                reasoning_parts.append(f"Key changes: {'; '.join(trend_notes[:3])}.")
            if observations:
                reasoning_parts.append(f"Concerning findings: {', '.join(observations)}.")
            if abs_delta > 0.2:
                reasoning_parts.append(
                    "This represents a significant revision. Prior assessment underestimated "
                    "emerging risk — the new evidence substantially changes the clinical picture."
                )
        else:
            # Risk decreasing
            reasoning_parts.append(
                f"Revising downward from {prev_str} to {risk:.2f}."
            )
            if tx_events:
                # The critical de-anchoring signal: improvement under active
                # treatment is *response to therapy*, not "I was wrong about sepsis."
                reasoning_parts.append(
                    f"Patient responding to treatment: {'; '.join(tx_events)}. "
                    "Improvement reflects effective intervention, not spontaneous recovery — "
                    "the underlying sepsis risk remains, but the trajectory is favorable."
                )
            elif (current_row.get("TX_VASO_ANY") or 0) > 0 or current_row.get("TX_VENT_INVASIVE"):
                # Still on active treatment — improvement is treatment-mediated
                reasoning_parts.append(
                    "Improvement occurring under ongoing treatment (vasopressors/ventilation "
                    "still active). Risk reduction reflects therapeutic response, not resolution."
                )
            if trend_notes:
                reasoning_parts.append(f"Improving trends: {'; '.join(trend_notes[:3])}.")
            if not tx_events and not (
                (current_row.get("TX_VASO_ANY") or 0) > 0 or current_row.get("TX_VENT_INVASIVE")
            ):
                reasoning_parts.append("Clinical trajectory suggests improvement.")

    reasoning = " ".join(reasoning_parts)

    return (
        f"RISK: {risk:.2f}\n"
        f"PREVIOUS: {prev_str}\n"
        f"DELTA: {delta_str}\n"
        f"REASONING: {reasoning}"
    )


def build_expert_trace(patient_hours_sofa, step_interval=2):
    """
    Build a full multi-turn expert trace for one patient.

    Args:
        patient_hours_sofa: DataFrame with HOUR, SOFA, and vital/lab columns.
        step_interval: Include every Nth hour to keep sequences manageable.

    Returns:
        List of messages [{role, content}, ...] for the full trajectory.
    """
    df = patient_hours_sofa.sort_values("HOUR").reset_index(drop=True)
    hours = sorted(df["HOUR"].unique())

    # Sample hours at interval, always include first and last
    sampled_hours = [hours[0]]
    sampled_hours += [h for h in hours[1:-1] if h % step_interval == 0]
    if hours[-1] not in sampled_hours:
        sampled_hours.append(hours[-1])

    messages = [{"role": "system", "content": TRAJECTORY_SYSTEM_PROMPT}]

    prev_risk = None
    prev_row = None

    for h in sampled_hours:
        row = df[df["HOUR"] == h].iloc[0]
        sofa_val = row.get("SOFA", 0)
        risk = sofa_to_risk_label(sofa_val)

        # User turn: new hour's observations
        obs_text = format_hour_observations(row)
        user_msg = f"Hour {int(h)} vitals/labs:\n{obs_text}\n\nUpdate your sepsis risk assessment."
        messages.append({"role": "user", "content": user_msg})

        # Assistant turn: expert assessment with explicit revision
        assistant_msg = generate_expert_response(risk, prev_risk, h, row, prev_row, df)
        messages.append({"role": "assistant", "content": assistant_msg})

        prev_risk = risk
        prev_row = row

    return messages, sampled_hours


def prepare_sft_dataset(config_path="config/paths.yaml"):
    """
    Create trajectory-aware SFT training dataset from processed MIMIC-III data.

    Each example is a full multi-turn expert trace through one patient's
    ICU stay, teaching the model to revise beliefs as evidence evolves.
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

    # Track which patients have sepsis for balanced reporting
    sepsis_ids = set(sepsis["ICUSTAY_ID"].values)

    logger.info(f"Patients — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    logger.info(f"Sepsis patients: {len(sepsis_ids)}")

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        examples = []
        n_sepsis = 0
        total_turns = 0

        for icustay_id in split_ids:
            patient = hourly_sofa[hourly_sofa["ICUSTAY_ID"] == icustay_id]
            if len(patient) < 4:  # need enough hours for a meaningful trajectory
                continue

            messages, sampled_hours = build_expert_trace(patient, step_interval=2)
            has_sepsis = icustay_id in sepsis_ids

            example = {
                "messages": messages,
                "icustay_id": int(icustay_id),
                "has_sepsis": has_sepsis,
                "n_turns": len(sampled_hours),
                "hours_covered": [int(h) for h in sampled_hours],
            }
            examples.append(example)
            total_turns += len(sampled_hours)
            if has_sepsis:
                n_sepsis += 1

        # Save
        outpath = out_dir / f"{split_name}.jsonl"
        with open(outpath, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        logger.info(
            f"{split_name}: {len(examples)} traces ({n_sepsis} sepsis), "
            f"{total_turns} total turns -> {outpath}"
        )

    logger.info("Trajectory-aware SFT dataset preparation complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_sft_dataset()
