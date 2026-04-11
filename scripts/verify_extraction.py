"""Quick sanity check after 01_extract_cohort completes."""
import pandas as pd

h = pd.read_parquet("data/processed/hourly_timelines.parquet")
tx = [c for c in h.columns if c.startswith("TX_")]
print("TX columns:", tx)
print("Rows with any pressor:", (h["TX_VASO_ANY"] > 0).sum())
print("Unique stays:", h["ICUSTAY_ID"].nunique())

s = pd.read_parquet("data/processed/sepsis_onset.parquet")
print("Sepsis onset patients:", len(s))
