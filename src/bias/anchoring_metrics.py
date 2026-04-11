"""
Phase 4: Anchoring bias measurement metrics.

Implements the three-part innovation:
1. Counterfactual Ordering Effect
2. Belief Update Elasticity
3. Explanation Drift
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)


# ============================================================
# 1. COUNTERFACTUAL ORDERING EFFECT
# ============================================================

def load_predictions(filepath):
    """Load JSONL predictions file into DataFrame."""
    records = []
    with open(filepath) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_ordering_effect(chrono_path, reverse_path, shuffled_path=None):
    """
    Compare final predictions across orderings for the same patients.

    If early information dominates, chronological and reverse orderings
    will produce systematically different final predictions despite
    containing identical evidence.

    Returns:
        DataFrame with per-patient ordering effect metrics.
    """
    chrono = load_predictions(chrono_path)
    reverse = load_predictions(reverse_path)

    # Get final-hour prediction per patient per ordering
    def get_final_predictions(df):
        return (
            df.sort_values("hour")
            .groupby("icustay_id")
            .last()
            [["risk", "has_sepsis", "sepsis_onset_hour"]]
            .reset_index()
        )

    chrono_final = get_final_predictions(chrono).rename(columns={"risk": "risk_chrono"})
    reverse_final = get_final_predictions(reverse).rename(columns={"risk": "risk_reverse"})

    merged = chrono_final.merge(reverse_final[["icustay_id", "risk_reverse"]], on="icustay_id")
    merged["ordering_effect"] = (merged["risk_chrono"] - merged["risk_reverse"]).abs()

    # For patients who develop sepsis but start stable:
    # chrono should underpredict (anchored on early stability)
    # reverse should overpredict (anchored on early deterioration)
    sepsis_patients = merged[merged["has_sepsis"]].dropna(subset=["risk_chrono", "risk_reverse"])
    if len(sepsis_patients) > 0:
        # Directional test: is chrono systematically lower?
        t_stat, p_val = stats.ttest_rel(
            sepsis_patients["risk_chrono"],
            sepsis_patients["risk_reverse"],
        )
        logger.info(f"Paired t-test (sepsis patients): t={t_stat:.3f}, p={p_val:.4f}")
        logger.info(f"Mean chrono risk: {sepsis_patients['risk_chrono'].mean():.3f}")
        logger.info(f"Mean reverse risk: {sepsis_patients['risk_reverse'].mean():.3f}")

    if shuffled_path:
        shuffled = load_predictions(shuffled_path)
        shuffled_final = get_final_predictions(shuffled).rename(columns={"risk": "risk_shuffled"})
        merged = merged.merge(shuffled_final[["icustay_id", "risk_shuffled"]], on="icustay_id")

    logger.info(f"Mean ordering effect: {merged['ordering_effect'].mean():.4f}")
    logger.info(f"Median ordering effect: {merged['ordering_effect'].median():.4f}")

    return merged


# ============================================================
# 2. BELIEF UPDATE ELASTICITY
# ============================================================

def compute_elasticity(predictions_path):
    """
    Measure how much the model's risk estimate changes at each timestep
    in response to new evidence.

    Elasticity = |risk(t) - risk(t-1)| / |evidence_change(t)|

    Low elasticity = model is anchored (doesn't update despite new evidence).
    High elasticity = model is responsive to new information.

    Returns:
        DataFrame with per-patient, per-hour elasticity scores.
    """
    preds = load_predictions(predictions_path)
    preds = preds.sort_values(["icustay_id", "hour"])
    preds = preds.dropna(subset=["risk"])

    results = []
    for icustay_id, group in preds.groupby("icustay_id"):
        group = group.sort_values("hour").reset_index(drop=True)
        for i in range(1, len(group)):
            prev_risk = group.loc[i - 1, "risk"]
            curr_risk = group.loc[i, "risk"]
            delta_risk = curr_risk - prev_risk
            abs_delta = abs(delta_risk)

            results.append({
                "icustay_id": icustay_id,
                "hour": group.loc[i, "hour"],
                "prev_risk": prev_risk,
                "curr_risk": curr_risk,
                "delta_risk": delta_risk,
                "abs_delta_risk": abs_delta,
                "has_sepsis": group.loc[i, "has_sepsis"],
                "sepsis_onset_hour": group.loc[i, "sepsis_onset_hour"],
            })

    df = pd.DataFrame(results)

    # Key metric: elasticity around sepsis onset
    sepsis = df[df["has_sepsis"] & df["sepsis_onset_hour"].notna()].copy()
    if len(sepsis) > 0:
        sepsis["hours_to_onset"] = sepsis["hour"] - sepsis["sepsis_onset_hour"]
        # Pre-onset (should be increasing but anchored model won't)
        pre_onset = sepsis[(sepsis["hours_to_onset"] >= -6) & (sepsis["hours_to_onset"] < 0)]
        post_onset = sepsis[(sepsis["hours_to_onset"] >= 0) & (sepsis["hours_to_onset"] <= 6)]

        logger.info(f"Mean |delta_risk| pre-onset (6h): {pre_onset['abs_delta_risk'].mean():.4f}")
        logger.info(f"Mean |delta_risk| post-onset (6h): {post_onset['abs_delta_risk'].mean():.4f}")

    return df


def compute_treatment_responsive_elasticity(predictions_path, hourly_path):
    """
    Treatment-conditioned belief update elasticity.

    The standard `compute_elasticity` measures |delta_risk| at every hour
    indiscriminately. This function joins predictions to the hourly
    timelines (which carry TX_* treatment columns) and stratifies the
    elasticity by what kind of evidence the model just received:

      - post_treatment_event: hours immediately following a vasopressor
        start/stop/escalation/wean or intubation/extubation.
      - baseline (vitals_only): hours with no treatment transition.
      - asymmetric: |delta_risk| on hours where evidence is *worsening*
        (treatment escalation, intubation) vs *improving* (treatment
        wean, extubation).

    Anchored models show:
      - Suppressed elasticity post-treatment events overall.
      - Asymmetric elasticity: they update on worsening evidence
        ("aha, more sepsis") but not on improving evidence
        ("the patient is responding") — the anchoring signature.

    Ordering caveat:
        This metric assumes predictions were produced under CHRONOLOGICAL
        inference. The function joins predictions back to chronological
        hours and computes deltas between consecutive clinical hours, so:
          - Chronological runs:  measures incremental update response. PRIMARY.
          - Reverse runs:        measures de-accumulation response (what the
                                 model thought after seeing 72->h+1 vs 72->h).
                                 The numbers are coherent but answer a
                                 different question. Use only as a paired
                                 comparison against chronological.
          - Shuffled runs:       not meaningful. Consecutive clinical hours
                                 were not seen consecutively by the model,
                                 so post-event deltas reflect nothing causal.
                                 Do not report this metric on shuffled runs.

    Args:
        predictions_path: JSONL of model risk traces (icustay_id, hour, risk).
        hourly_path: parquet from extract_cohort.py with TX_* columns.

    Returns:
        dict of summary metrics.
    """
    preds = load_predictions(predictions_path)
    if "risk" not in preds.columns or len(preds) == 0:
        logger.warning(f"No usable predictions in {predictions_path}")
        return {}

    hourly = pd.read_parquet(hourly_path)
    tx_cols = [c for c in hourly.columns if c.startswith("TX_")]
    if not tx_cols:
        logger.warning(f"No TX_* columns in {hourly_path} — "
                       "treatment-conditioned elasticity requires the v2 cohort.")
        return {}

    # Normalize join keys
    hourly = hourly.rename(columns={"ICUSTAY_ID": "icustay_id", "HOUR": "hour"})
    hourly["icustay_id"] = hourly["icustay_id"].astype(int)
    hourly["hour"] = hourly["hour"].astype(int)
    preds["icustay_id"] = preds["icustay_id"].astype(int)
    preds["hour"] = preds["hour"].astype(int)

    merged = preds.merge(
        hourly[["icustay_id", "hour"] + tx_cols],
        on=["icustay_id", "hour"],
        how="left",
    )
    merged = merged.sort_values(["icustay_id", "hour"]).reset_index(drop=True)
    merged = merged.dropna(subset=["risk"])

    rows = []
    for icustay_id, group in merged.groupby("icustay_id"):
        group = group.sort_values("hour").reset_index(drop=True)
        for i in range(1, len(group)):
            prev = group.loc[i - 1]
            curr = group.loc[i]
            delta = curr["risk"] - prev["risk"]
            abs_delta = abs(delta)

            # Detect treatment transitions between t-1 and t
            event_type = None  # "worsening", "improving", or None
            prev_norepi = prev.get("TX_NOREPI_RATE", 0) or 0
            curr_norepi = curr.get("TX_NOREPI_RATE", 0) or 0
            prev_vent = prev.get("TX_VENT_INVASIVE", 0) or 0
            curr_vent = curr.get("TX_VENT_INVASIVE", 0) or 0

            if prev_norepi == 0 and curr_norepi > 0:
                event_type = "worsening"  # pressor started
            elif prev_norepi > 0 and curr_norepi == 0:
                event_type = "improving"  # pressor weaned off
            elif curr_norepi > prev_norepi * 1.5 and prev_norepi > 0:
                event_type = "worsening"  # escalation
            elif curr_norepi > 0 and curr_norepi < prev_norepi * 0.5:
                event_type = "improving"  # weaning
            elif not prev_vent and curr_vent:
                event_type = "worsening"  # intubation
            elif prev_vent and not curr_vent:
                event_type = "improving"  # extubation

            rows.append({
                "icustay_id": int(icustay_id),
                "hour": int(curr["hour"]),
                "abs_delta_risk": abs_delta,
                "delta_risk": delta,
                "event_type": event_type,
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {}

    baseline = df[df["event_type"].isnull()]
    post_event = df[df["event_type"].notnull()]
    worsening = df[df["event_type"] == "worsening"]
    improving = df[df["event_type"] == "improving"]

    summary = {
        "n_baseline_hours": int(len(baseline)),
        "n_treatment_event_hours": int(len(post_event)),
        "n_worsening_events": int(len(worsening)),
        "n_improving_events": int(len(improving)),
        "mean_elasticity_baseline": float(baseline["abs_delta_risk"].mean()) if len(baseline) else None,
        "mean_elasticity_post_treatment": float(post_event["abs_delta_risk"].mean()) if len(post_event) else None,
        "mean_elasticity_worsening": float(worsening["abs_delta_risk"].mean()) if len(worsening) else None,
        "mean_elasticity_improving": float(improving["abs_delta_risk"].mean()) if len(improving) else None,
    }
    # Asymmetry ratio: anchored models show this >> 1
    if summary["mean_elasticity_improving"] and summary["mean_elasticity_improving"] > 0:
        summary["asymmetry_ratio"] = (
            summary["mean_elasticity_worsening"] / summary["mean_elasticity_improving"]
        )
    else:
        summary["asymmetry_ratio"] = None

    # Signed responsiveness on improving events: a non-anchored model
    # should produce *negative* delta_risk on improving events. An anchored
    # model produces small or even positive deltas (still going up despite
    # the patient responding).
    if len(improving) > 0:
        summary["mean_signed_delta_on_improving"] = float(improving["delta_risk"].mean())
    else:
        summary["mean_signed_delta_on_improving"] = None

    logger.info(f"Treatment-responsive elasticity: {summary}")
    return summary


def compute_bayesian_ideal(hourly_path, sepsis_path):
    """
    Compute a Bayesian ideal updater's risk trajectory for comparison.

    Uses SOFA trends to compute what an optimal updater would predict,
    serving as a benchmark for elasticity.
    """
    hourly = pd.read_parquet(hourly_path)
    sepsis = pd.read_parquet(sepsis_path)

    # Simple Bayesian baseline: risk proportional to SOFA percentile
    # This is a rough benchmark — the point is to show the model
    # updates LESS than even a simple statistical baseline
    sofa_cols = ["MAP", "PLATELETS", "BILIRUBIN", "CREATININE", "GCS"]
    available = [c for c in sofa_cols if c in hourly.columns]

    # Normalize each feature to [0,1] risk contribution
    risk_scores = []
    for icustay_id, group in hourly.groupby("ICUSTAY_ID"):
        group = group.sort_values("HOUR")
        # Simple: use z-score of deterioration features
        risk = pd.Series(0.0, index=group.index)
        if "MAP" in group.columns:
            risk += (70 - group["MAP"].fillna(70)).clip(lower=0) / 70
        if "LACTATE" in group.columns:
            risk += (group["LACTATE"].fillna(1) - 1).clip(lower=0) / 10
        if "CREATININE" in group.columns:
            risk += (group["CREATININE"].fillna(1) - 1).clip(lower=0) / 5

        risk = (risk / max(len(available), 1)).clip(0, 1)
        for h, r in zip(group["HOUR"], risk):
            risk_scores.append({
                "icustay_id": icustay_id,
                "hour": h,
                "bayesian_risk": r,
            })

    return pd.DataFrame(risk_scores)


# ============================================================
# 3. EXPLANATION DRIFT
# ============================================================

def compute_explanation_drift(predictions_path, model_name="all-MiniLM-L6-v2"):
    """
    Measure semantic similarity between early and late explanations.

    Anchored models retrofit early reasoning to match late predictions
    (low drift in explanation despite high drift in evidence).
    Non-anchored models genuinely revise their reasoning.

    Returns:
        DataFrame with per-patient explanation drift metrics.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping explanation drift")
        return pd.DataFrame()

    preds = load_predictions(predictions_path)
    preds = preds.dropna(subset=["reasoning"])

    model = SentenceTransformer(model_name)

    results = []
    for icustay_id, group in preds.groupby("icustay_id"):
        group = group.sort_values("hour").reset_index(drop=True)
        if len(group) < 4:
            continue

        reasonings = group["reasoning"].tolist()
        embeddings = model.encode(reasonings)

        # Compare each explanation to the first one
        first_emb = embeddings[0:1]
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(first_emb, embeddings[i:i+1])[0][0]
            results.append({
                "icustay_id": icustay_id,
                "hour": group.loc[i, "hour"],
                "similarity_to_first": sim,
                "drift_from_first": 1 - sim,
                "has_sepsis": group.loc[i, "has_sepsis"],
            })

    return pd.DataFrame(results)


def run_analysis_for_mode(results_dir, model_tag, mode_label, chrono_path, reverse_path,
                          shuffled_path=None, hourly_path=None):
    """Run all analyses for a given inference mode (single-turn or multi-turn)."""
    print(f"\n{'=' * 60}")
    print(f"ANCHORING BIAS ANALYSIS — {mode_label}")
    print(f"{'=' * 60}")

    if not chrono_path.exists() or not reverse_path.exists():
        print(f"  Skipping {mode_label}: missing prediction files")
        return

    # 1. Ordering effect
    print(f"\n--- Counterfactual Ordering Effect ({mode_label}) ---")
    ordering_df = compute_ordering_effect(
        chrono_path, reverse_path,
        shuffled_path if shuffled_path and shuffled_path.exists() else None,
    )
    ordering_df.to_csv(results_dir / f"ordering_effect_{mode_label}.csv", index=False)

    # 2. Elasticity (both orderings)
    for ordering, path in [("chronological", chrono_path), ("reverse", reverse_path)]:
        print(f"\n--- Belief Update Elasticity ({mode_label}, {ordering}) ---")
        elasticity_df = compute_elasticity(path)
        elasticity_df.to_csv(
            results_dir / f"elasticity_{ordering}_{mode_label}.csv", index=False
        )

    # 3. Explanation drift (both orderings)
    for ordering, path in [("chronological", chrono_path), ("reverse", reverse_path)]:
        print(f"\n--- Explanation Drift ({mode_label}, {ordering}) ---")
        drift_df = compute_explanation_drift(path)
        if len(drift_df) > 0:
            drift_df.to_csv(
                results_dir / f"explanation_drift_{ordering}_{mode_label}.csv", index=False
            )

    # 4. Treatment-responsive elasticity (chronological only — see docstring)
    if hourly_path is not None and Path(hourly_path).exists():
        print(f"\n--- Treatment-Responsive Elasticity ({mode_label}, chronological) ---")
        tx_summary = compute_treatment_responsive_elasticity(str(chrono_path), str(hourly_path))
        if tx_summary:
            import json
            out = results_dir / f"treatment_elasticity_{mode_label}.json"
            with open(out, "w") as f:
                json.dump(tx_summary, f, indent=2)
            print(f"  Saved to {out.name}")
            for k, v in tx_summary.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print("  No treatment events found.")

    print(f"\n{mode_label} analysis complete.")


def run_full_analysis(config_path="config/paths.yaml"):
    """Run all anchoring bias metrics for single-turn and multi-turn modes."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["results_dir"])
    processed_dir = Path(cfg["processed_dir"])
    model_tag = cfg["ollama"]["model_base"].replace(":", "_")
    hourly_path = processed_dir / "hourly_timelines.parquet"

    # Single-turn analysis
    run_analysis_for_mode(
        results_dir, model_tag, "single_turn",
        chrono_path=results_dir / f"predictions_chronological_single_turn_{model_tag}.jsonl",
        reverse_path=results_dir / f"predictions_reverse_single_turn_{model_tag}.jsonl",
        shuffled_path=results_dir / f"predictions_shuffled_single_turn_{model_tag}.jsonl",
        hourly_path=hourly_path,
    )

    # Multi-turn analysis
    run_analysis_for_mode(
        results_dir, model_tag, "multi_turn",
        chrono_path=results_dir / f"predictions_chronological_multiturn_{model_tag}.jsonl",
        reverse_path=results_dir / f"predictions_reverse_multiturn_{model_tag}.jsonl",
        shuffled_path=results_dir / f"predictions_shuffled_multiturn_{model_tag}.jsonl",
        hourly_path=hourly_path,
    )

    print("\nAll analyses complete. Results saved to", results_dir)


if __name__ == "__main__":
    run_full_analysis()
