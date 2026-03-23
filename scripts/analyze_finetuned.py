"""Generate elasticity, explanation drift, and ordering effect CSVs for fine-tuned model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bias.anchoring_metrics import (
    compute_ordering_effect,
    compute_elasticity,
    compute_explanation_drift,
)


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/results")

    chrono = results_dir / "predictions_chronological_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"
    reverse = results_dir / "predictions_reverse_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"

    suffix = "finetuned_multi_turn"

    # Ordering effect
    if chrono.exists() and reverse.exists():
        print(f"--- Ordering Effect ({suffix}) ---")
        oe = compute_ordering_effect(chrono, reverse)
        out = results_dir / f"ordering_effect_{suffix}.csv"
        oe.to_csv(out, index=False)
        print(f"  Saved: {out.name}  (n={len(oe)})")
        print(f"  Mean ordering effect: {oe['ordering_effect'].mean():.4f}")

    # Elasticity (both orderings)
    for ordering, path in [("chronological", chrono), ("reverse", reverse)]:
        if not path.exists():
            print(f"  Skipping elasticity {ordering}: file not found")
            continue
        print(f"\n--- Elasticity ({suffix}, {ordering}) ---")
        el = compute_elasticity(path)
        out = results_dir / f"elasticity_{ordering}_{suffix}.csv"
        el.to_csv(out, index=False)
        print(f"  Saved: {out.name}  (n={len(el)})")
        print(f"  Mean |delta_risk|: {el['abs_delta_risk'].mean():.4f}")

    # Explanation drift (both orderings)
    for ordering, path in [("chronological", chrono), ("reverse", reverse)]:
        if not path.exists():
            print(f"  Skipping drift {ordering}: file not found")
            continue
        print(f"\n--- Explanation Drift ({suffix}, {ordering}) ---")
        dr = compute_explanation_drift(path)
        if len(dr) > 0:
            out = results_dir / f"explanation_drift_{ordering}_{suffix}.csv"
            dr.to_csv(out, index=False)
            print(f"  Saved: {out.name}  (n={len(dr)})")
            print(f"  Mean drift: {dr['drift_from_first'].mean():.4f}")
            print(f"  Mean similarity: {dr['similarity_to_first'].mean():.4f}")
        else:
            print(f"  No explanation drift data (missing reasoning field?)")

    print("\nDone.")


if __name__ == "__main__":
    main()
