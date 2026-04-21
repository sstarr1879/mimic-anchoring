"""Overfit/ordering diagnostic given incomplete fine-tuned prediction files.

Fine-tuned reverse run crashed (71 lines, 3 patients) and shuffled is partial
(1,758 lines, ~66 patients). This script does the best comparison possible:
  1. chrono_ft vs shuffled_ft on the patients that exist in both
  2. chrono_ft vs reverse_ft on the 3 reverse patients (sanity only)
  3. the same patients under the pre-SFT baseline (v1/) to check whether
     SFT increased or decreased ordering sensitivity on that cohort.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


RESULTS = Path("results")
V1 = RESULTS / "v1"


def load(p):
    rows = []
    with open(p) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def final_per_patient(df):
    return (
        df.sort_values("hour")
        .groupby("icustay_id")
        .last()[["risk", "has_sepsis", "sepsis_onset_hour"]]
        .reset_index()
    )


def describe(tag, df):
    print(f"  {tag}: rows={len(df)}  patients={df['icustay_id'].nunique()}  "
          f"hours/patient(median)={df.groupby('icustay_id').size().median():.0f}")


def pairwise(a, b, label_a, label_b):
    m = a.merge(b, on="icustay_id", suffixes=(f"_{label_a}", f"_{label_b}"))
    diff = (m[f"risk_{label_a}"] - m[f"risk_{label_b}"]).abs()
    print(f"  {label_a} vs {label_b}: n={len(m)}  "
          f"mean |diff|={diff.mean():.4f}  median={diff.median():.4f}")
    if len(m) >= 5:
        t, p = stats.ttest_rel(m[f"risk_{label_a}"], m[f"risk_{label_b}"])
        print(f"    paired t-test: t={t:.3f}  p={p:.4f}  "
              f"mean({label_a})={m[f'risk_{label_a}'].mean():.3f}  "
              f"mean({label_b})={m[f'risk_{label_b}'].mean():.3f}")
    return m


def main():
    ft_chr = RESULTS / "predictions_chronological_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"
    ft_rev = RESULTS / "predictions_reverse_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"
    ft_shf = RESULTS / "predictions_shuffled_multiturn_llama3.1-sepsis_8b_finetuned.jsonl"
    bl_chr = V1 / "predictions_chronological_multiturn_llama3.1_8b.jsonl"
    bl_rev = V1 / "predictions_reverse_multiturn_llama3.1_8b.jsonl"

    print("=" * 70)
    print("FILE INVENTORY")
    print("=" * 70)
    ft_chr_df = load(ft_chr); describe("finetuned chrono   ", ft_chr_df)
    ft_rev_df = load(ft_rev); describe("finetuned reverse  ", ft_rev_df)
    ft_shf_df = load(ft_shf); describe("finetuned shuffled ", ft_shf_df)
    bl_chr_df = load(bl_chr); describe("baseline  chrono   ", bl_chr_df)
    bl_rev_df = load(bl_rev); describe("baseline  reverse  ", bl_rev_df)

    # Final-hour per patient (this is what "ordering effect" compares)
    ft_chr_f = final_per_patient(ft_chr_df).rename(columns={"risk": "risk"})
    ft_rev_f = final_per_patient(ft_rev_df)
    ft_shf_f = final_per_patient(ft_shf_df)
    bl_chr_f = final_per_patient(bl_chr_df)
    bl_rev_f = final_per_patient(bl_rev_df)

    # Restrict comparisons to patients present in shuffled (the binding constraint)
    shf_ids = set(ft_shf_f["icustay_id"])
    rev_ids = set(ft_rev_f["icustay_id"])
    print(f"\n  shuffled patient ids: n={len(shf_ids)}")
    print(f"  reverse  patient ids: n={len(rev_ids)}")
    print(f"  overlap shuffled ∩ chrono(ft): "
          f"{len(shf_ids & set(ft_chr_f['icustay_id']))}")
    print(f"  overlap reverse  ∩ chrono(ft): "
          f"{len(rev_ids & set(ft_chr_f['icustay_id']))}")
    print(f"  overlap reverse  ∩ shuffled:   {len(rev_ids & shf_ids)}")

    print("\n" + "=" * 70)
    print("ORDERING EFFECT — FINE-TUNED MODEL (what we can measure)")
    print("=" * 70)
    # chrono vs shuffled on shuffled's patients
    a = ft_chr_f[["icustay_id", "risk"]].rename(columns={"risk": "risk_chrono"})
    b = ft_shf_f[["icustay_id", "risk"]].rename(columns={"risk": "risk_shuffled"})
    m_cs = a.merge(b, on="icustay_id")
    m_cs["diff"] = (m_cs["risk_chrono"] - m_cs["risk_shuffled"]).abs()
    print(f"\n  chrono(ft) vs shuffled(ft): n={len(m_cs)}")
    print(f"    mean |diff|   = {m_cs['diff'].mean():.4f}")
    print(f"    median |diff| = {m_cs['diff'].median():.4f}")
    print(f"    stdev         = {m_cs['diff'].std():.4f}")
    if len(m_cs) >= 5:
        t, p = stats.ttest_rel(m_cs["risk_chrono"], m_cs["risk_shuffled"])
        print(f"    paired t: t={t:.3f}  p={p:.4f}  "
              f"mean(chr)={m_cs['risk_chrono'].mean():.3f}  "
              f"mean(shf)={m_cs['risk_shuffled'].mean():.3f}")
    m_cs.to_csv(RESULTS / "ordering_effect_finetuned_chrono_shuffled.csv", index=False)
    print(f"    saved: ordering_effect_finetuned_chrono_shuffled.csv")

    # chrono vs reverse (fine-tuned) — full pairwise
    c = ft_rev_f[["icustay_id", "risk"]].rename(columns={"risk": "risk_reverse"})
    m_cr = a.merge(c, on="icustay_id").dropna(subset=["risk_chrono", "risk_reverse"])
    m_cr["diff"] = (m_cr["risk_chrono"] - m_cr["risk_reverse"]).abs()
    print(f"\n  chrono(ft) vs reverse(ft): n={len(m_cr)}")
    print(f"    mean |diff|   = {m_cr['diff'].mean():.4f}")
    print(f"    median |diff| = {m_cr['diff'].median():.4f}")
    if len(m_cr) >= 5:
        t, p = stats.ttest_rel(m_cr["risk_chrono"], m_cr["risk_reverse"])
        print(f"    paired t: t={t:.3f}  p={p:.4f}  "
              f"mean(chr)={m_cr['risk_chrono'].mean():.3f}  "
              f"mean(rev)={m_cr['risk_reverse'].mean():.3f}")

    # reverse vs shuffled (fine-tuned) — THE overfit check
    br = ft_rev_f[["icustay_id", "risk"]].rename(columns={"risk": "risk_reverse"})
    bs = ft_shf_f[["icustay_id", "risk"]].rename(columns={"risk": "risk_shuffled"})
    m_rs = br.merge(bs, on="icustay_id").dropna(subset=["risk_reverse", "risk_shuffled"])
    m_rs["diff"] = (m_rs["risk_reverse"] - m_rs["risk_shuffled"]).abs()
    print(f"\n  reverse(ft) vs shuffled(ft) [overfit check]: n={len(m_rs)}")
    print(f"    mean |diff|   = {m_rs['diff'].mean():.4f}")
    print(f"    median |diff| = {m_rs['diff'].median():.4f}")
    if len(m_rs) >= 5:
        t, p = stats.ttest_rel(m_rs["risk_reverse"], m_rs["risk_shuffled"])
        print(f"    paired t: t={t:.3f}  p={p:.4f}  "
              f"mean(rev)={m_rs['risk_reverse'].mean():.3f}  "
              f"mean(shf)={m_rs['risk_shuffled'].mean():.3f}")
    m_rs.to_csv(RESULTS / "ordering_effect_finetuned_reverse_shuffled.csv", index=False)
    print(f"    saved: ordering_effect_finetuned_reverse_shuffled.csv")

    # Combined 3-way table on the 308 patients present in all three
    abc = (
        a.merge(br, on="icustay_id")
         .merge(bs, on="icustay_id")
         .dropna(subset=["risk_chrono", "risk_reverse", "risk_shuffled"])
    )
    abc["d_cr"] = (abc["risk_chrono"] - abc["risk_reverse"]).abs()
    abc["d_cs"] = (abc["risk_chrono"] - abc["risk_shuffled"]).abs()
    abc["d_rs"] = (abc["risk_reverse"] - abc["risk_shuffled"]).abs()
    print(f"\n  3-way (all orderings) overlap: n={len(abc)}")
    print(f"    mean |chrono-reverse|    = {abc['d_cr'].mean():.4f}")
    print(f"    mean |chrono-shuffled|   = {abc['d_cs'].mean():.4f}")
    print(f"    mean |reverse-shuffled|  = {abc['d_rs'].mean():.4f}")
    print("    Overfit-to-reverse signature: |rev-shf| >> |chr-shf|.")
    print("    Generalized de-anchoring:     all three should be small and similar.")
    abc.to_csv(RESULTS / "ordering_effect_finetuned_3way.csv", index=False)
    print(f"    saved: ordering_effect_finetuned_3way.csv")

    print("\n" + "=" * 70)
    print("SAME PATIENTS UNDER PRE-SFT BASELINE (v1/)")
    print("=" * 70)
    print("  Goal: did SFT reduce the chrono↔other ordering gap on this cohort,")
    print("  or widen it? If SFT widens chrono↔shuffled, fine-tuning isn't")
    print("  generalizing the de-anchoring — it's a reverse-pattern artifact.\n")

    # Baseline on the shuffled cohort's patients (chrono vs reverse, full baseline files)
    baseline_ids = shf_ids & set(bl_chr_f["icustay_id"]) & set(bl_rev_f["icustay_id"])
    ba = bl_chr_f[bl_chr_f["icustay_id"].isin(baseline_ids)][["icustay_id", "risk"]].rename(
        columns={"risk": "risk_chr_bl"}
    )
    bb = bl_rev_f[bl_rev_f["icustay_id"].isin(baseline_ids)][["icustay_id", "risk"]].rename(
        columns={"risk": "risk_rev_bl"}
    )
    bm = ba.merge(bb, on="icustay_id")
    bm["diff_bl"] = (bm["risk_chr_bl"] - bm["risk_rev_bl"]).abs()
    print(f"  baseline chrono vs baseline reverse on shuffled-cohort patients: n={len(bm)}")
    print(f"    mean |chr_bl - rev_bl|   = {bm['diff_bl'].mean():.4f}")
    print(f"    median                   = {bm['diff_bl'].median():.4f}")

    # Baseline chrono vs fine-tuned chrono (same patients)
    m_bl_ft = ba.merge(
        ft_chr_f[["icustay_id", "risk"]].rename(columns={"risk": "risk_chr_ft"}),
        on="icustay_id",
    )
    m_bl_ft["diff_train"] = (m_bl_ft["risk_chr_ft"] - m_bl_ft["risk_chr_bl"]).abs()
    print(f"\n  chrono baseline vs chrono finetuned: n={len(m_bl_ft)}")
    print(f"    mean shift = {m_bl_ft['diff_train'].mean():.4f}  "
          f"(how much SFT changed chrono predictions on these patients)")
    if len(m_bl_ft) >= 5:
        t, p = stats.ttest_rel(m_bl_ft["risk_chr_ft"], m_bl_ft["risk_chr_bl"])
        print(f"    paired t: t={t:.3f}  p={p:.4f}  "
              f"mean(ft)={m_bl_ft['risk_chr_ft'].mean():.3f}  "
              f"mean(bl)={m_bl_ft['risk_chr_bl'].mean():.3f}")

    print("\n" + "=" * 70)
    print("ELASTICITY COMPARISON")
    print("=" * 70)

    def elasticity(df, tag):
        df = df.sort_values(["icustay_id", "hour"]).dropna(subset=["risk"])
        deltas = (
            df.groupby("icustay_id")["risk"]
            .apply(lambda s: s.diff().abs().dropna())
        )
        mean_abs = deltas.mean()
        print(f"  {tag}: n_transitions={len(deltas)}  "
              f"mean |delta_risk|={mean_abs:.4f}  "
              f"frac_exactly_zero={(deltas == 0).mean():.3f}")
        return mean_abs

    print("\n  Fine-tuned:")
    elasticity(ft_chr_df, "chrono   (ft)         ")
    elasticity(ft_rev_df, "reverse  (ft)         ")
    elasticity(ft_shf_df, "shuffled (ft)         ")

    print("\n  Baseline (v1, same patients as shuffled-ft):")
    bl_chr_sub = bl_chr_df[bl_chr_df["icustay_id"].isin(shf_ids)]
    bl_rev_sub = bl_rev_df[bl_rev_df["icustay_id"].isin(shf_ids)]
    elasticity(bl_chr_sub, "chrono   (bl, subset) ")
    elasticity(bl_rev_sub, "reverse  (bl, subset) ")

    print("\n  If ft chrono and ft shuffled have similar |delta_risk|, the")
    print("  model updates comparably in both orderings. If shuffled is much")
    print("  flatter, SFT taught ordering-specific update behavior.")

    print("\nDone.")


if __name__ == "__main__":
    main()
