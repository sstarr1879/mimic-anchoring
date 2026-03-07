"""
Phase 1a: Extract sepsis cohort from MIMIC-III CSVs using Sepsis-3 criteria.

Sepsis-3 definition (Singer et al., JAMA 2016):
  - Suspected infection: antibiotics ordered AND microbiology culture taken
  - Organ dysfunction: SOFA score >= 2

Output: cohort dataframe with icustay_id, sepsis_onset_time, and metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config/paths.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_table(mimic_dir, table_name, usecols=None, dtype=None):
    """Load a MIMIC-III CSV (handles .csv and .csv.gz)."""
    path = Path(mimic_dir)
    for ext in [f"{table_name}.csv.gz", f"{table_name}.csv"]:
        fpath = path / ext
        if fpath.exists():
            logger.info(f"Loading {fpath}")
            return pd.read_csv(fpath, usecols=usecols, dtype=dtype, low_memory=False)
    raise FileNotFoundError(f"{table_name} not found in {mimic_dir}")


def get_suspected_infection(mimic_dir):
    """
    Identify suspected infection events:
    - Antibiotics prescribed within 24h of microbiology culture (or vice versa).
    Returns DataFrame with HADM_ID, ICUSTAY_ID, SUSPECTED_INFECTION_TIME.
    """
    # Load prescriptions for antibiotics
    prescriptions = load_table(mimic_dir, "PRESCRIPTIONS",
                               usecols=["HADM_ID", "ICUSTAY_ID", "STARTDATE", "DRUG"])
    # Common antibiotic keywords
    abx_keywords = [
        "cillin", "cephalos", "mycin", "oxacin", "azole", "cycline",
        "sulfa", "trim", "vancomycin", "meropenem", "imipenem",
        "aztreonam", "metronidazole", "daptomycin", "linezolid",
        "ceftriaxone", "cefepime", "piperacillin", "tazobactam",
        "ciprofloxacin", "levofloxacin", "amoxicillin", "ampicillin",
        "gentamicin", "tobramycin", "clindamycin", "azithromycin",
    ]
    mask = prescriptions["DRUG"].str.lower().str.contains(
        "|".join(abx_keywords), na=False
    )
    abx = prescriptions[mask].copy()
    abx["ABX_TIME"] = pd.to_datetime(abx["STARTDATE"], errors="coerce")
    abx = abx.dropna(subset=["ABX_TIME", "HADM_ID"])

    # Load microbiology cultures
    micro = load_table(mimic_dir, "MICROBIOLOGYEVENTS",
                       usecols=["HADM_ID", "CHARTTIME"])
    micro["CULTURE_TIME"] = pd.to_datetime(micro["CHARTTIME"], errors="coerce")
    micro = micro.dropna(subset=["CULTURE_TIME", "HADM_ID"])

    # Merge on HADM_ID, find pairs within 24h
    merged = abx.merge(micro[["HADM_ID", "CULTURE_TIME"]], on="HADM_ID")
    merged["TIME_DIFF_H"] = (
        (merged["ABX_TIME"] - merged["CULTURE_TIME"]).dt.total_seconds() / 3600
    ).abs()
    merged = merged[merged["TIME_DIFF_H"] <= 24]

    # Suspected infection time = earlier of abx or culture
    merged["SUSPECTED_INFECTION_TIME"] = merged[["ABX_TIME", "CULTURE_TIME"]].min(axis=1)

    # Take earliest per admission
    result = (
        merged.groupby("HADM_ID")["SUSPECTED_INFECTION_TIME"]
        .min()
        .reset_index()
    )
    logger.info(f"Found {len(result)} admissions with suspected infection")
    return result


def compute_sofa_components(mimic_dir, icustays):
    """
    Compute hourly SOFA scores from chartevents and labevents.
    Simplified: uses available vitals/labs to approximate SOFA.

    Returns DataFrame with ICUSTAY_ID, HOUR, SOFA_SCORE and component details.
    """
    # SOFA-relevant item IDs from MIMIC-III
    # Respiration: PaO2/FiO2 ratio
    # Coagulation: Platelets
    # Liver: Bilirubin
    # Cardiovascular: MAP + vasopressors
    # CNS: GCS
    # Renal: Creatinine + urine output

    vital_items = {
        # MAP
        456: "MAP", 52: "MAP", 6702: "MAP", 443: "MAP", 220052: "MAP", 220181: "MAP",
        # GCS
        198: "GCS",
        # Heart rate
        211: "HR", 220045: "HR",
        # Temperature
        223761: "TEMP_F", 678: "TEMP_F", 223762: "TEMP_C", 676: "TEMP_C",
        # SpO2
        646: "SPO2", 220277: "SPO2",
        # Respiratory rate
        618: "RR", 220210: "RR", 224690: "RR",
    }

    lab_items = {
        50821: "PO2",         # PaO2
        50816: "FIO2",        # FiO2
        51265: "PLATELETS",
        50885: "BILIRUBIN",
        50912: "CREATININE",
        51301: "WBC",
        50813: "LACTATE",
    }

    icustay_ids = set(icustays["ICUSTAY_ID"].dropna().astype(int))

    # Load chartevents (large — filter by itemid)
    logger.info("Loading CHARTEVENTS (this may take a while)...")
    charts = load_table(
        mimic_dir, "CHARTEVENTS",
        usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
        dtype={"ICUSTAY_ID": "float64", "ITEMID": "int64", "VALUENUM": "float64"},
    )
    charts = charts[charts["ITEMID"].isin(vital_items.keys())]
    charts = charts[charts["ICUSTAY_ID"].isin(icustay_ids)]
    charts["CHARTTIME"] = pd.to_datetime(charts["CHARTTIME"], errors="coerce")
    charts["LABEL"] = charts["ITEMID"].map(vital_items)
    charts = charts.dropna(subset=["CHARTTIME", "VALUENUM", "ICUSTAY_ID"])
    charts["ICUSTAY_ID"] = charts["ICUSTAY_ID"].astype(int)

    # Load labevents
    logger.info("Loading LABEVENTS...")
    labs = load_table(
        mimic_dir, "LABEVENTS",
        usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
        dtype={"HADM_ID": "float64", "ITEMID": "int64", "VALUENUM": "float64"},
    )
    labs = labs[labs["ITEMID"].isin(lab_items.keys())]
    labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"], errors="coerce")
    labs["LABEL"] = labs["ITEMID"].map(lab_items)
    labs = labs.dropna(subset=["CHARTTIME", "VALUENUM"])

    # Map labs to icustay via hadm_id
    hadm_to_icu = icustays[["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]].copy()
    hadm_to_icu["INTIME"] = pd.to_datetime(hadm_to_icu["INTIME"])
    hadm_to_icu["OUTTIME"] = pd.to_datetime(hadm_to_icu["OUTTIME"])
    labs = labs.merge(hadm_to_icu, on="HADM_ID")
    labs = labs[
        (labs["CHARTTIME"] >= labs["INTIME"]) & (labs["CHARTTIME"] <= labs["OUTTIME"])
    ]
    labs["ICUSTAY_ID"] = labs["ICUSTAY_ID"].astype(int)

    # Combine vitals and labs
    vitals_df = charts[["ICUSTAY_ID", "CHARTTIME", "LABEL", "VALUENUM"]].copy()
    labs_df = labs[["ICUSTAY_ID", "CHARTTIME", "LABEL", "VALUENUM"]].copy()
    all_obs = pd.concat([vitals_df, labs_df], ignore_index=True)

    logger.info(f"Total observations: {len(all_obs):,}")
    return all_obs


def build_hourly_timelines(observations, icustays, max_hours=72, bin_minutes=60):
    """
    Bin observations into hourly windows relative to ICU admission.
    Returns pivoted DataFrame: ICUSTAY_ID x HOUR x features.
    """
    icu = icustays[["ICUSTAY_ID", "INTIME"]].copy()
    icu["INTIME"] = pd.to_datetime(icu["INTIME"])

    obs = observations.merge(icu, on="ICUSTAY_ID")
    obs["HOURS_SINCE_ADMIT"] = (
        (obs["CHARTTIME"] - obs["INTIME"]).dt.total_seconds() / 3600
    )
    # Filter to valid time range
    obs = obs[(obs["HOURS_SINCE_ADMIT"] >= 0) & (obs["HOURS_SINCE_ADMIT"] < max_hours)]
    obs["HOUR"] = (obs["HOURS_SINCE_ADMIT"] // (bin_minutes / 60)).astype(int)

    # Pivot: mean value per hour per feature
    hourly = (
        obs.groupby(["ICUSTAY_ID", "HOUR", "LABEL"])["VALUENUM"]
        .mean()
        .reset_index()
        .pivot_table(index=["ICUSTAY_ID", "HOUR"], columns="LABEL", values="VALUENUM")
        .reset_index()
    )
    hourly.columns.name = None

    logger.info(f"Built hourly timelines: {hourly['ICUSTAY_ID'].nunique()} stays, "
                f"{len(hourly)} total rows")
    return hourly


def compute_sofa_from_hourly(hourly):
    """Approximate SOFA score from available hourly features."""
    sofa = pd.DataFrame(index=hourly.index)
    sofa["ICUSTAY_ID"] = hourly["ICUSTAY_ID"]
    sofa["HOUR"] = hourly["HOUR"]
    sofa["SOFA"] = 0

    # Cardiovascular: MAP
    if "MAP" in hourly.columns:
        sofa["SOFA"] += np.where(hourly["MAP"] < 70, 1, 0)

    # Respiration: PaO2/FiO2
    if "PO2" in hourly.columns and "FIO2" in hourly.columns:
        ratio = hourly["PO2"] / hourly["FIO2"].clip(lower=0.21)
        sofa["SOFA"] += np.select(
            [ratio < 100, ratio < 200, ratio < 300, ratio < 400],
            [4, 3, 2, 1], default=0
        )

    # Coagulation: Platelets
    if "PLATELETS" in hourly.columns:
        sofa["SOFA"] += np.select(
            [hourly["PLATELETS"] < 20, hourly["PLATELETS"] < 50,
             hourly["PLATELETS"] < 100, hourly["PLATELETS"] < 150],
            [4, 3, 2, 1], default=0
        )

    # Liver: Bilirubin
    if "BILIRUBIN" in hourly.columns:
        sofa["SOFA"] += np.select(
            [hourly["BILIRUBIN"] >= 12, hourly["BILIRUBIN"] >= 6,
             hourly["BILIRUBIN"] >= 2, hourly["BILIRUBIN"] >= 1.2],
            [4, 3, 2, 1], default=0
        )

    # CNS: GCS
    if "GCS" in hourly.columns:
        sofa["SOFA"] += np.select(
            [hourly["GCS"] < 6, hourly["GCS"] < 10,
             hourly["GCS"] < 13, hourly["GCS"] < 15],
            [4, 3, 2, 1], default=0
        )

    # Renal: Creatinine
    if "CREATININE" in hourly.columns:
        sofa["SOFA"] += np.select(
            [hourly["CREATININE"] >= 5, hourly["CREATININE"] >= 3.5,
             hourly["CREATININE"] >= 2, hourly["CREATININE"] >= 1.2],
            [4, 3, 2, 1], default=0
        )

    return sofa


def extract_sepsis_cohort(config_path="config/paths.yaml"):
    """Main pipeline: extract sepsis cohort from MIMIC-III."""
    cfg = load_config(config_path)
    mimic_dir = cfg["mimic_raw_dir"]
    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ICU stays
    icustays = load_table(mimic_dir, "ICUSTAYS")
    icustays["INTIME"] = pd.to_datetime(icustays["INTIME"])
    icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"])
    icustays["LOS_HOURS"] = (icustays["OUTTIME"] - icustays["INTIME"]).dt.total_seconds() / 3600
    icustays = icustays[icustays["LOS_HOURS"] >= cfg["cohort"]["min_icu_hours"]]
    logger.info(f"ICU stays >= {cfg['cohort']['min_icu_hours']}h: {len(icustays)}")

    # 2. Get suspected infection
    infection = get_suspected_infection(mimic_dir)
    cohort = icustays.merge(infection, on="HADM_ID")
    # Infection must occur during ICU stay (or within 24h before)
    cohort = cohort[
        cohort["SUSPECTED_INFECTION_TIME"] >= (cohort["INTIME"] - pd.Timedelta(hours=24))
    ]
    logger.info(f"ICU stays with suspected infection: {len(cohort)}")

    # 3. Extract observations and build timelines
    observations = compute_sofa_components(mimic_dir, cohort)
    hourly = build_hourly_timelines(
        observations, cohort,
        max_hours=cfg["cohort"]["max_hours"],
        bin_minutes=cfg["cohort"]["bin_window_minutes"],
    )

    # 4. Compute SOFA and identify sepsis onset
    sofa = compute_sofa_from_hourly(hourly)
    sepsis_onset = (
        sofa[sofa["SOFA"] >= cfg["cohort"]["sofa_threshold"]]
        .groupby("ICUSTAY_ID")["HOUR"]
        .min()
        .reset_index()
        .rename(columns={"HOUR": "SEPSIS_ONSET_HOUR"})
    )
    logger.info(f"Patients meeting Sepsis-3 (SOFA >= {cfg['cohort']['sofa_threshold']}): "
                f"{len(sepsis_onset)}")

    # 5. Save outputs
    hourly.to_parquet(out_dir / "hourly_timelines.parquet", index=False)
    sepsis_onset.to_parquet(out_dir / "sepsis_onset.parquet", index=False)
    cohort.to_parquet(out_dir / "cohort_metadata.parquet", index=False)
    sofa.to_parquet(out_dir / "sofa_scores.parquet", index=False)

    logger.info(f"Saved processed data to {out_dir}")
    return hourly, sepsis_onset, cohort, sofa


if __name__ == "__main__":
    extract_sepsis_cohort()
