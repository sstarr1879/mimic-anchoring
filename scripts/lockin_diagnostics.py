"""
Trajectory Lock-In Diagnostics: Baseline vs Fine-Tuned

Runs the four key tests that distinguish trajectory lock-in from classical anchoring:
1. First-risk vs final-risk correlation (anchoring test)
2. Responsiveness decay by quartile (lock-in signature)
3. Output collapse / default value clustering
4. Explanation drift comparison

Usage:
    python scripts/lockin_diagnostics.py [results_dir]
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats


def load_predictions(filepath):
    """Load JSONL predictions into DataFrame."""
    records = []
    with open(filepath) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def get_first_and_final(df):
    """Get first-hour and final-hour risk per patient."""
    df = df.dropna(subset=["risk"]).sort_values(["icustay_id", "hour"])
    first = df.groupby("icustay_id").first()[["risk"]].rename(columns={"risk": "first_risk"})
    final = df.groupby("icustay_id").last()[["risk"]].rename(columns={"risk": "final_risk"})
    return first.join(final)


def test_anchoring(df, label):
    """Test 1: Does first risk predict final risk? (Classical anchoring = yes)"""
    ff = get_first_and_final(df)
    ff = ff.dropna()
    r, p = stats.pearsonr(ff["first_risk"], ff["final_risk"])

    print(f"\n  First-risk vs final-risk correlation:")
    print(f"    r = {r:.3f}, p = {p:.4f}")
    print(f"    Mean first risk:  {ff['first_risk'].mean():.3f}")
    print(f"    Mean final risk:  {ff['final_risk'].mean():.3f}")
    print(f"    Mean shift:       {(ff['final_risk'] - ff['first_risk']).mean():+.3f}")

    if abs(r) < 0.1:
        print(f"    >> First impression has NO predictive power (NOT classical anchoring)")
    elif abs(r) > 0.5:
        print(f"    >> First impression PREDICTS final risk (consistent with anchoring)")
    else:
        print(f"    >> Weak relationship between first and final risk")

    return {"r": r, "p": p, "mean_first": ff["first_risk"].mean(),
            "mean_final": ff["final_risk"].mean()}


def test_responsiveness_decay(df, label):
    """Test 2: Does responsiveness decay over time? (Lock-in signature)"""
    df = df.dropna(subset=["risk"]).sort_values(["icustay_id", "hour"])

    deltas = []
    for pid, group in df.groupby("icustay_id"):
        group = group.sort_values("hour").reset_index(drop=True)
        n = len(group)
        if n < 4:
            continue
        for i in range(1, n):
            quartile = int((i / n) * 4)
            quartile = min(quartile, 3)  # 0-3
            deltas.append({
                "quartile": quartile + 1,  # 1-4
                "abs_delta": abs(group.loc[i, "risk"] - group.loc[i-1, "risk"]),
                "is_zero": abs(group.loc[i, "risk"] - group.loc[i-1, "risk"]) < 0.001,
            })

    delta_df = pd.DataFrame(deltas)

    print(f"\n  Responsiveness by quartile of stay:")
    print(f"    {'Quartile':<12} {'Mean |delta|':>14} {'Zero-update %':>15}")
    print(f"    {'-'*12} {'-'*14} {'-'*15}")

    q_means = []
    for q in [1, 2, 3, 4]:
        qd = delta_df[delta_df["quartile"] == q]
        m = qd["abs_delta"].mean()
        z = 100 * qd["is_zero"].mean()
        q_means.append(m)
        print(f"    Q{q:<11} {m:>14.4f} {z:>14.1f}%")

    if q_means[0] > 0:
        decay = 100 * (1 - q_means[3] / q_means[0])
        print(f"\n    Responsiveness decay (Q1 to Q4): {decay:+.1f}%")
        if decay > 50:
            print(f"    >> STRONG lock-in: model stops updating in late stay")
        elif decay > 25:
            print(f"    >> MODERATE lock-in: responsiveness drops over time")
        elif decay < -25:
            print(f"    >> INVERSE pattern: model becomes MORE responsive over time")
        else:
            print(f"    >> MINIMAL decay: model maintains responsiveness (lock-in mitigated)")

    return {"q1_mean": q_means[0], "q4_mean": q_means[3]}


def test_output_collapse(df, label):
    """Test 3: Does the model collapse to default output values?"""
    df = df.dropna(subset=["risk"]).sort_values(["icustay_id", "hour"])
    final = df.groupby("icustay_id").last()["risk"]

    # Round to 2 decimal places and find most common values
    rounded = final.round(2)
    counts = Counter(rounded)
    top5 = counts.most_common(5)
    n = len(final)

    print(f"\n  Output collapse analysis (final risk distribution):")
    print(f"    Total patients: {n}")
    print(f"    Unique final risk values: {len(counts)}")
    print(f"\n    Most common final risks:")
    collapse_pct = 0
    for val, count in top5:
        pct = 100 * count / n
        if pct > 10:
            collapse_pct += pct
        print(f"      {val:.2f}: {count} patients ({pct:.1f}%)")

    print(f"\n    Std dev of final risks: {final.std():.3f}")
    print(f"    IQR: [{final.quantile(0.25):.2f}, {final.quantile(0.75):.2f}]")

    if collapse_pct > 50:
        print(f"    >> OUTPUT COLLAPSE: {collapse_pct:.0f}% of patients cluster at default values")
    elif collapse_pct > 25:
        print(f"    >> PARTIAL COLLAPSE: some clustering at default values")
    elif len(counts) > n * 0.3:
        print(f"    >> DIVERSE outputs: model produces patient-specific predictions")
    else:
        print(f"    >> Moderate output diversity")

    return {"unique_values": len(counts), "top_value": top5[0][0],
            "top_pct": 100 * top5[0][1] / n, "std": final.std()}


def test_explanation_drift(results_dir, ordering, model_label, suffix):
    """Test 4: Does the model recycle reasoning or genuinely revise?"""
    # Look for pre-computed drift CSVs first
    drift_file = results_dir / f"explanation_drift_{ordering}_{suffix}.csv"
    if not drift_file.exists():
        # Try without suffix
        drift_file = results_dir / f"explanation_drift_{ordering}.csv"
    if not drift_file.exists():
        return None

    df = pd.read_csv(drift_file)
    mean_drift = df["drift_from_first"].mean()
    mean_sim = df["similarity_to_first"].mean()

    print(f"\n  Explanation drift ({ordering}):")
    print(f"    Mean drift from first explanation: {mean_drift:.3f}")
    print(f"    Mean similarity to first:          {mean_sim:.3f}")

    if mean_sim > 0.8:
        print(f"    >> NARRATIVE RIGIDITY: model recycles same reasoning frame")
    elif mean_sim < 0.6:
        print(f"    >> GENUINE REVISION: model substantially revises reasoning")
    else:
        print(f"    >> MODERATE revision in reasoning")

    return {"mean_drift": mean_drift, "mean_similarity": mean_sim}


def run_diagnostics(results_dir, chrono_path, reverse_path, label, drift_suffix=None):
    """Run all four lock-in diagnostics on a pair of prediction files."""
    print(f"\n{'=' * 70}")
    print(f"  TRAJECTORY LOCK-IN DIAGNOSTICS: {label}")
    print(f"{'=' * 70}")

    for ordering, path in [("chronological", chrono_path), ("reverse", reverse_path)]:
        if not path.exists():
            print(f"\n  Skipping {ordering}: {path.name} not found")
            continue

        df = load_predictions(path)
        print(f"\n{'─' * 70}")
        print(f"  [{ordering.upper()}] ({path.name})")
        print(f"  Patients: {df['icustay_id'].nunique()}, Hours: {len(df)}")
        print(f"{'─' * 70}")

        test_anchoring(df, f"{label} {ordering}")
        test_responsiveness_decay(df, f"{label} {ordering}")
        test_output_collapse(df, f"{label} {ordering}")

        if drift_suffix:
            test_explanation_drift(results_dir, ordering, label, drift_suffix)


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")

    # --- Baseline multi-turn ---
    bc = results_dir / "predictions_chronological_multiturn_llama3.1_8b.jsonl"
    br = results_dir / "predictions_reverse_multiturn_llama3.1_8b.jsonl"
    if bc.exists() and br.exists():
        run_diagnostics(results_dir, bc, br,
                        "BASELINE (zero-shot, multi-turn)",
                        drift_suffix="multi_turn")

    # --- Fine-tuned multi-turn ---
    fc = results_dir / "predictions_chronological_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"
    fr = results_dir / "predictions_reverse_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"
    if fc.exists() and fr.exists():
        run_diagnostics(results_dir, fc, fr,
                        "FINE-TUNED (LoRA SFT, multi-turn)",
                        drift_suffix="finetuned_multi_turn")

    # --- Side-by-side comparison ---
    if bc.exists() and br.exists() and fc.exists() and fr.exists():
        print(f"\n{'=' * 70}")
        print(f"  BASELINE vs FINE-TUNED COMPARISON")
        print(f"{'=' * 70}")

        for ordering, bp, fp in [("chronological", bc, fc), ("reverse", br, fr)]:
            b_df = load_predictions(bp)
            f_df = load_predictions(fp)
            b_ff = get_first_and_final(b_df).dropna()
            f_ff = get_first_and_final(f_df).dropna()

            b_r, _ = stats.pearsonr(b_ff["first_risk"], b_ff["final_risk"])
            f_r, _ = stats.pearsonr(f_ff["first_risk"], f_ff["final_risk"])

            b_final_std = b_df.dropna(subset=["risk"]).groupby("icustay_id").last()["risk"].std()
            f_final_std = f_df.dropna(subset=["risk"]).groupby("icustay_id").last()["risk"].std()

            print(f"\n  {ordering.upper()}:")
            print(f"    {'Metric':<35} {'Baseline':>10} {'Fine-tuned':>12}")
            print(f"    {'-'*35} {'-'*10} {'-'*12}")
            print(f"    {'First→Final correlation (r)':<35} {b_r:>10.3f} {f_r:>12.3f}")
            print(f"    {'Mean first risk':<35} {b_ff['first_risk'].mean():>10.3f} {f_ff['first_risk'].mean():>12.3f}")
            print(f"    {'Mean final risk':<35} {b_ff['final_risk'].mean():>10.3f} {f_ff['final_risk'].mean():>12.3f}")
            print(f"    {'Final risk std dev':<35} {b_final_std:>10.3f} {f_final_std:>12.3f}")

    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSTICS COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
