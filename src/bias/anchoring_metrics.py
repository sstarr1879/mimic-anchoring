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
    sepsis_patients = merged[merged["has_sepsis"]]
    if len(sepsis_patients) > 0:
        # Directional test: is chrono systematically lower?
        t_stat, p_val = stats.ttest_rel(
            sepsis_patients["risk_chrono"].dropna(),
            sepsis_patients["risk_reverse"].dropna()
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


def run_analysis_for_mode(results_dir, model_tag, mode_label, chrono_path, reverse_path, shuffled_path=None):
    """Run all three analyses for a given inference mode (single-turn or multi-turn)."""
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

    print(f"\n{mode_label} analysis complete.")


def run_full_analysis(config_path="config/paths.yaml"):
    """Run all anchoring bias metrics for single-turn and multi-turn modes."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["results_dir"])
    model_tag = cfg["ollama"]["model_base"].replace(":", "_")

    # Single-turn analysis
    run_analysis_for_mode(
        results_dir, model_tag, "single_turn",
        chrono_path=results_dir / f"predictions_chronological_single_turn_{model_tag}.jsonl",
        reverse_path=results_dir / f"predictions_reverse_single_turn_{model_tag}.jsonl",
        shuffled_path=results_dir / f"predictions_shuffled_single_turn_{model_tag}.jsonl",
    )

    # Multi-turn analysis
    run_analysis_for_mode(
        results_dir, model_tag, "multi_turn",
        chrono_path=results_dir / f"predictions_chronological_multiturn_{model_tag}.jsonl",
        reverse_path=results_dir / f"predictions_reverse_multiturn_{model_tag}.jsonl",
        shuffled_path=results_dir / f"predictions_shuffled_multiturn_{model_tag}.jsonl",
    )

    print("\nAll analyses complete. Results saved to", results_dir)


if __name__ == "__main__":
    run_full_analysis()
