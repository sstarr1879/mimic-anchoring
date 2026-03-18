"""Run Belief Update Elasticity and Explanation Drift analyses on completed predictions."""
import logging
import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.bias.anchoring_metrics import compute_elasticity, compute_explanation_drift

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")

# Find prediction files
chrono_files = list(results_dir.glob("predictions_chronological_*single_turn*.jsonl"))
reverse_files = list(results_dir.glob("predictions_reverse_*single_turn*.jsonl"))

if not chrono_files:
    print("No chronological prediction files found in", results_dir)
    sys.exit(1)

# === Belief Update Elasticity ===
print("=" * 50)
print("BELIEF UPDATE ELASTICITY")
print("=" * 50)

for label, files in [("chronological", chrono_files), ("reverse", reverse_files)]:
    if not files:
        print(f"\nSkipping {label} (no file found)")
        continue
    filepath = files[0]
    print(f"\n--- {label}: {filepath.name} ---")
    df = compute_elasticity(str(filepath))
    outpath = results_dir / f"elasticity_{label}.csv"
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} rows to {outpath.name}")

    # Summary stats
    print(f"  Mean |delta_risk| per step: {df['abs_delta_risk'].mean():.4f}")
    print(f"  Median |delta_risk| per step: {df['abs_delta_risk'].median():.4f}")
    print(f"  Patients with near-zero updates (<0.01): "
          f"{(df.groupby('icustay_id')['abs_delta_risk'].mean() < 0.01).sum()}")

# === Explanation Drift ===
print("\n" + "=" * 50)
print("EXPLANATION DRIFT")
print("=" * 50)

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    for label, files in [("chronological", chrono_files), ("reverse", reverse_files)]:
        if not files:
            continue
        filepath = files[0]
        print(f"\n--- {label}: {filepath.name} ---")
        df = compute_explanation_drift(str(filepath))
        if len(df) > 0:
            outpath = results_dir / f"explanation_drift_{label}.csv"
            df.to_csv(outpath, index=False)
            print(f"Saved {len(df)} rows to {outpath.name}")
            print(f"  Mean drift from first explanation: {df['drift_from_first'].mean():.4f}")
            print(f"  Mean similarity to first: {df['similarity_to_first'].mean():.4f}")
        else:
            print("  No explanation drift data (missing 'reasoning' field?)")
except ImportError:
    print("\nsentence-transformers not installed, skipping explanation drift.")
    print("Install with: pip install sentence-transformers")
