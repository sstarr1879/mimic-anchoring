"""Quick analysis of baseline inference results to check for anchoring effect."""
import json
import sys
from pathlib import Path
import numpy as np

def load_predictions(filepath):
    records = []
    with open(filepath) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def get_final_risk(records):
    """Get final-hour risk per patient."""
    patients = {}
    for r in records:
        pid = r["icustay_id"]
        if pid not in patients or r["hour"] > patients[pid]["hour"]:
            patients[pid] = r
    return patients

def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")

    # Find prediction files
    chrono_files = list(results_dir.glob("predictions_chronological_*multiturn*.jsonl"))
    reverse_files = list(results_dir.glob("predictions_reverse_*multiturn*.jsonl"))

    if not chrono_files or not reverse_files:
        # Try single_turn
        chrono_files = list(results_dir.glob("predictions_chronological_*single_turn*.jsonl"))
        reverse_files = list(results_dir.glob("predictions_reverse_*single_turn*.jsonl"))

    if not chrono_files or not reverse_files:
        print("No prediction files found in", results_dir)
        print("Available files:", [f.name for f in results_dir.glob("*.jsonl")])
        return

    print(f"Using: {chrono_files[0].name}")
    print(f"   vs: {reverse_files[0].name}")

    chrono = get_final_risk(load_predictions(chrono_files[0]))
    reverse = get_final_risk(load_predictions(reverse_files[0]))

    # Match patients
    shared = set(chrono.keys()) & set(reverse.keys())
    print(f"\nPatients compared: {len(shared)}")

    chrono_risks = []
    reverse_risks = []
    diffs = []
    sepsis_chrono = []
    sepsis_reverse = []

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

    chrono_risks = np.array(chrono_risks)
    reverse_risks = np.array(reverse_risks)
    diffs = np.array(diffs)

    print(f"Patients with valid risk scores: {len(diffs)}")

    print("\n" + "=" * 50)
    print("ANCHORING EFFECT SUMMARY")
    print("=" * 50)

    print(f"\nOverall mean risk (chronological): {chrono_risks.mean():.3f}")
    print(f"Overall mean risk (reverse):       {reverse_risks.mean():.3f}")
    print(f"Mean absolute difference:          {diffs.mean():.3f}")
    print(f"Median absolute difference:        {np.median(diffs):.3f}")
    print(f"Patients with >10% difference:     {(diffs > 0.10).sum()} / {len(diffs)} ({100*(diffs > 0.10).mean():.1f}%)")
    print(f"Patients with >20% difference:     {(diffs > 0.20).sum()} / {len(diffs)} ({100*(diffs > 0.20).mean():.1f}%)")

    if sepsis_chrono:
        sc = np.array(sepsis_chrono)
        sr = np.array(sepsis_reverse)
        print(f"\n--- Sepsis patients only (n={len(sc)}) ---")
        print(f"Mean risk (chronological): {sc.mean():.3f}")
        print(f"Mean risk (reverse):       {sr.mean():.3f}")
        print(f"Difference:                {sc.mean() - sr.mean():.3f}")
        if sc.mean() < sr.mean():
            print(">> Chronological UNDERPREDICTS vs reverse = anchoring on early stability")
        else:
            print(">> Reverse UNDERPREDICTS vs chronological")

    # Simple significance test
    if len(diffs) > 1:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(chrono_risks, reverse_risks)
        print(f"\nPaired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(">> SIGNIFICANT anchoring effect detected (p < 0.05)")
        else:
            print(">> No significant anchoring effect (p >= 0.05)")

if __name__ == "__main__":
    main()
