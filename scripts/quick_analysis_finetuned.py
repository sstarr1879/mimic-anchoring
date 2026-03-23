"""Quick comparison of anchoring effect: baseline vs fine-tuned model."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats


def load_final_risks(filepath):
    """Load JSONL and return dict of {patient_id: record} for final hour."""
    patients = {}
    with open(filepath) as f:
        for line in f:
            r = json.loads(line)
            pid = r["icustay_id"]
            if pid not in patients or r["hour"] > patients[pid]["hour"]:
                patients[pid] = r
    return patients


def analyze_pair(chrono_path, reverse_path, label):
    """Run anchoring analysis on a chrono/reverse pair."""
    chrono = load_final_risks(chrono_path)
    reverse = load_final_risks(reverse_path)

    shared = set(chrono.keys()) & set(reverse.keys())

    chrono_risks, reverse_risks, diffs = [], [], []
    sepsis_chrono, sepsis_reverse = [], []

    for pid in shared:
        cr = chrono[pid].get("risk")
        rr = reverse[pid].get("risk")
        if cr is None or rr is None:
            continue
        chrono_risks.append(cr)
        reverse_risks.append(rr)
        diffs.append(abs(cr - rr))
        if chrono[pid].get("has_sepsis"):
            sepsis_chrono.append(cr)
            sepsis_reverse.append(rr)

    cr = np.array(chrono_risks)
    rr = np.array(reverse_risks)
    diffs = np.array(diffs)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Files: {chrono_path.name}")
    print(f"     vs: {reverse_path.name}")
    print(f"  Patients with valid scores: {len(diffs)}")
    print(f"\n  Mean risk (chronological):   {cr.mean():.3f}")
    print(f"  Mean risk (reverse):         {rr.mean():.3f}")
    print(f"  Mean absolute difference:    {diffs.mean():.3f}")
    print(f"  Median absolute difference:  {np.median(diffs):.3f}")
    print(f"  Patients with >10% diff:     {(diffs > 0.10).sum()} / {len(diffs)} ({100*(diffs > 0.10).mean():.1f}%)")
    print(f"  Patients with >20% diff:     {(diffs > 0.20).sum()} / {len(diffs)} ({100*(diffs > 0.20).mean():.1f}%)")

    if sepsis_chrono:
        sc, sr = np.array(sepsis_chrono), np.array(sepsis_reverse)
        print(f"\n  --- Sepsis patients only (n={len(sc)}) ---")
        print(f"  Mean risk (chronological):   {sc.mean():.3f}")
        print(f"  Mean risk (reverse):         {sr.mean():.3f}")
        print(f"  Difference:                  {sc.mean() - sr.mean():.3f}")

    if len(diffs) > 1:
        t_stat, p_val = stats.ttest_rel(cr, rr)
        print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("  >> SIGNIFICANT anchoring effect (p < 0.05)")
        else:
            print("  >> No significant anchoring effect (p >= 0.05)")

    return {"label": label, "n": len(diffs), "mean_diff": diffs.mean(),
            "median_diff": np.median(diffs), "pct_gt10": 100*(diffs > 0.10).mean(),
            "pct_gt20": 100*(diffs > 0.20).mean(),
            "mean_chrono": cr.mean(), "mean_reverse": rr.mean()}


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")

    pairs = []

    # Baseline multi-turn
    bc = results_dir / "predictions_chronological_multiturn_llama3.1_8b.jsonl"
    br = results_dir / "predictions_reverse_multiturn_llama3.1_8b.jsonl"
    if bc.exists() and br.exists():
        pairs.append((bc, br, "BASELINE (zero-shot, multi-turn)"))

    # Baseline single-turn
    bc_st = results_dir / "predictions_chronological_single_turn_llama3.1_8b.jsonl"
    br_st = results_dir / "predictions_reverse_single_turn_llama3.1_8b.jsonl"
    if bc_st.exists() and br_st.exists():
        pairs.append((bc_st, br_st, "BASELINE (zero-shot, single-turn)"))

    # Fine-tuned multi-turn
    fc = results_dir / "predictions_chronological_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"
    fr = results_dir / "predictions_reverse_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"
    if fc.exists() and fr.exists():
        pairs.append((fc, fr, "FINE-TUNED (LoRA SFT, multi-turn)"))

    if not pairs:
        print("No prediction file pairs found in", results_dir)
        print("Available:", [f.name for f in results_dir.glob("*.jsonl")])
        return

    summaries = []
    for chrono_path, reverse_path, label in pairs:
        summaries.append(analyze_pair(chrono_path, reverse_path, label))

    # Comparison table
    if len(summaries) > 1:
        print(f"\n{'=' * 60}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Model':<35} {'Mean Diff':>10} {'> 10%':>8} {'> 20%':>8}")
        print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8}")
        for s in summaries:
            print(f"  {s['label']:<35} {s['mean_diff']:>10.3f} {s['pct_gt10']:>7.1f}% {s['pct_gt20']:>7.1f}%")

        # Did fine-tuning help?
        baseline = [s for s in summaries if "BASELINE" in s["label"] and "multi-turn" in s["label"]]
        finetuned = [s for s in summaries if "FINE-TUNED" in s["label"]]
        if baseline and finetuned:
            b, f = baseline[0], finetuned[0]
            reduction = b["mean_diff"] - f["mean_diff"]
            pct_reduction = 100 * reduction / b["mean_diff"] if b["mean_diff"] > 0 else 0
            print(f"\n  Fine-tuning effect on anchoring bias:")
            print(f"    Baseline mean diff:   {b['mean_diff']:.3f}")
            print(f"    Fine-tuned mean diff: {f['mean_diff']:.3f}")
            print(f"    Reduction:            {reduction:+.3f} ({pct_reduction:+.1f}%)")
            if reduction > 0:
                print(f"    >> Fine-tuning REDUCED anchoring bias")
            elif reduction < 0:
                print(f"    >> Fine-tuning INCREASED anchoring bias")
            else:
                print(f"    >> No change in anchoring bias")


if __name__ == "__main__":
    main()
