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


# ============================================================
# Clinical treatment ITEMIDs (Metavision only)
# ============================================================
# These are distinct from the algorithmic "interventions" in
# src/interventions/ — these are *treatments delivered to the
# patient* (vasopressors, urine output, mechanical ventilation),
# extracted into TX_* columns on hourly_timelines.parquet.

VASOPRESSOR_ITEMS = {
    221906: "NOREPI",       # Norepinephrine
    221986: "VASOPRESSIN",  # Vasopressin
    221289: "EPI",          # Epinephrine
    221662: "DOPAMINE",     # Dopamine
    221749: "PHENYLEPH",    # Phenylephrine
}

URINE_OUTPUT_ITEMS = {
    226559: "FOLEY",
    226560: "VOID",
    226561: "CONDOM",
    226584: "ILEOCONDUIT",
    226563: "SUPRAPUBIC",
    226564: "RT_NEPHRO",
    226565: "LT_NEPHRO",
    227488: "GU_IRRIGANT_OUT",
    227489: "GU_IRRIGANT_IN",  # subtracted from total
}

VENT_ITEMS = {
    225792: "INVASIVE_VENT",     # Invasive ventilation (PROCEDUREEVENTS_MV)
    225794: "NONINVASIVE_VENT",  # Non-invasive ventilation
}


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

    # Load chartevents in chunks (too large to fit in memory at once)
    logger.info("Loading CHARTEVENTS in chunks (this may take a while)...")
    chart_path = Path(mimic_dir)
    for ext in [f"CHARTEVENTS.csv.gz", f"CHARTEVENTS.csv"]:
        fpath = chart_path / ext
        if fpath.exists():
            break
    else:
        raise FileNotFoundError("CHARTEVENTS not found in " + str(mimic_dir))
    chart_chunks = []
    for chunk in pd.read_csv(
        fpath,
        usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
        dtype={"ICUSTAY_ID": "float64", "ITEMID": "int64", "VALUENUM": "float64"},
        low_memory=False,
        chunksize=1_000_000,
    ):
        filtered = chunk[chunk["ITEMID"].isin(vital_items.keys())]
        filtered = filtered[filtered["ICUSTAY_ID"].isin(icustay_ids)]
        if len(filtered) > 0:
            chart_chunks.append(filtered)
    charts = pd.concat(chart_chunks, ignore_index=True)
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


def build_hourly_timelines(observations, icustays, max_hours=72, bin_minutes=60,
                           treatments=None):
    """
    Bin observations into hourly windows relative to ICU admission.
    Returns pivoted DataFrame: ICUSTAY_ID x HOUR x features.

    If `treatments` is provided (output of aggregate_treatments_hourly), its
    TX_* columns are left-joined onto the result. Missing TX values are
    filled with 0 (absence = treatment not active).
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

    if treatments is not None and len(treatments) > 0:
        tx_cols = [c for c in treatments.columns if c.startswith("TX_")]
        hourly = hourly.merge(
            treatments[["ICUSTAY_ID", "HOUR"] + tx_cols],
            on=["ICUSTAY_ID", "HOUR"],
            how="left",
        )
        for c in tx_cols:
            hourly[c] = hourly[c].fillna(0)

    logger.info(f"Built hourly timelines: {hourly['ICUSTAY_ID'].nunique()} stays, "
                f"{len(hourly)} total rows")
    return hourly


def _normalize_pressor_rate(rate, rateuom, weight_kg, drug):
    """
    Convert a vasopressor infusion rate to mcg/kg/min.

    MIMIC-III INPUTEVENTS_MV stores rates in a variety of units depending
    on the drug. The most common units we need to handle:
      - mcg/kg/min  → identity
      - mcg/min     → divide by weight_kg
      - mg/kg/min   → multiply by 1000
      - mg/min      → multiply by 1000, divide by weight_kg
      - units/min, units/hour → vasopressin only; convert to "rate equivalent"
        by leaving in original units (we report vasopressin presence, not
        a normalized rate, since the SOFA ladder uses dose-based cutoffs
        only for catecholamines).

    For vasopressin we return the raw rate (units/hr or units/min) and
    a separate boolean is used downstream. For all other pressors we
    return mcg/kg/min.
    """
    if pd.isna(rate) or rate is None:
        return np.nan
    if rateuom is None or pd.isna(rateuom):
        return np.nan
    uom = str(rateuom).lower().strip()
    weight = weight_kg if (weight_kg and weight_kg > 0) else 80.0  # fallback adult weight

    if drug == "VASOPRESSIN":
        # Don't normalize — vasopressin is dosed in units, presence is what matters
        return float(rate)

    if "mcg/kg/min" in uom:
        return float(rate)
    if "mcg/min" in uom:
        return float(rate) / weight
    if "mg/kg/min" in uom:
        return float(rate) * 1000.0
    if "mg/min" in uom:
        return float(rate) * 1000.0 / weight
    if "mg/kg/hour" in uom or "mg/kg/hr" in uom:
        return float(rate) * 1000.0 / 60.0
    if "mcg/kg/hour" in uom or "mcg/kg/hr" in uom:
        return float(rate) / 60.0
    return np.nan


def load_vasopressors(mimic_dir, icustay_ids):
    """
    Load vasopressor infusions from INPUTEVENTS_MV (Metavision only).

    Returns long-form DataFrame with one row per infusion segment:
      ICUSTAY_ID, STARTTIME, ENDTIME, DRUG, RATE_NORMALIZED
    where RATE_NORMALIZED is mcg/kg/min for catecholamines and raw
    units for vasopressin.
    """
    path = Path(mimic_dir)
    for ext in ["INPUTEVENTS_MV.csv.gz", "INPUTEVENTS_MV.csv"]:
        fpath = path / ext
        if fpath.exists():
            break
    else:
        logger.warning("INPUTEVENTS_MV not found — skipping vasopressors")
        return pd.DataFrame(columns=["ICUSTAY_ID", "STARTTIME", "ENDTIME", "DRUG", "RATE_NORMALIZED"])

    logger.info(f"Loading vasopressors from {fpath}")
    chunks = []
    for chunk in pd.read_csv(
        fpath,
        usecols=["ICUSTAY_ID", "ITEMID", "STARTTIME", "ENDTIME", "RATE", "RATEUOM", "PATIENTWEIGHT"],
        dtype={"ICUSTAY_ID": "float64", "ITEMID": "int64",
               "RATE": "float64", "PATIENTWEIGHT": "float64"},
        low_memory=False,
        chunksize=500_000,
    ):
        chunk = chunk[chunk["ITEMID"].isin(VASOPRESSOR_ITEMS.keys())]
        chunk = chunk[chunk["ICUSTAY_ID"].isin(icustay_ids)]
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=["ICUSTAY_ID", "STARTTIME", "ENDTIME", "DRUG", "RATE_NORMALIZED"])

    vaso = pd.concat(chunks, ignore_index=True)
    vaso["DRUG"] = vaso["ITEMID"].map(VASOPRESSOR_ITEMS)
    vaso["STARTTIME"] = pd.to_datetime(vaso["STARTTIME"], errors="coerce")
    vaso["ENDTIME"] = pd.to_datetime(vaso["ENDTIME"], errors="coerce")
    vaso = vaso.dropna(subset=["STARTTIME", "ENDTIME", "ICUSTAY_ID"])
    vaso["ICUSTAY_ID"] = vaso["ICUSTAY_ID"].astype(int)

    vaso["RATE_NORMALIZED"] = vaso.apply(
        lambda r: _normalize_pressor_rate(r["RATE"], r["RATEUOM"], r["PATIENTWEIGHT"], r["DRUG"]),
        axis=1,
    )
    logger.info(f"Loaded {len(vaso)} vasopressor infusion segments across "
                f"{vaso['ICUSTAY_ID'].nunique()} stays")
    return vaso[["ICUSTAY_ID", "STARTTIME", "ENDTIME", "DRUG", "RATE_NORMALIZED"]]


def load_urine_output(mimic_dir, icustay_ids):
    """
    Load urine output events from OUTPUTEVENTS.

    Returns DataFrame with ICUSTAY_ID, CHARTTIME, VOLUME_ML where
    VOLUME_ML is signed (positive for output, negative for irrigant in
    so the per-bin sum gives true urine output).
    """
    path = Path(mimic_dir)
    for ext in ["OUTPUTEVENTS.csv.gz", "OUTPUTEVENTS.csv"]:
        fpath = path / ext
        if fpath.exists():
            break
    else:
        logger.warning("OUTPUTEVENTS not found — skipping urine output")
        return pd.DataFrame(columns=["ICUSTAY_ID", "CHARTTIME", "VOLUME_ML"])

    logger.info(f"Loading urine output from {fpath}")
    chunks = []
    for chunk in pd.read_csv(
        fpath,
        usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUE"],
        dtype={"ICUSTAY_ID": "float64", "ITEMID": "int64", "VALUE": "float64"},
        low_memory=False,
        chunksize=500_000,
    ):
        chunk = chunk[chunk["ITEMID"].isin(URINE_OUTPUT_ITEMS.keys())]
        chunk = chunk[chunk["ICUSTAY_ID"].isin(icustay_ids)]
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=["ICUSTAY_ID", "CHARTTIME", "VOLUME_ML"])

    urine = pd.concat(chunks, ignore_index=True)
    urine["CHARTTIME"] = pd.to_datetime(urine["CHARTTIME"], errors="coerce")
    urine = urine.dropna(subset=["CHARTTIME", "VALUE", "ICUSTAY_ID"])
    urine["ICUSTAY_ID"] = urine["ICUSTAY_ID"].astype(int)
    # GU irrigant IN is subtracted from output to give true urine output
    urine["LABEL"] = urine["ITEMID"].map(URINE_OUTPUT_ITEMS)
    urine["VOLUME_ML"] = np.where(
        urine["LABEL"] == "GU_IRRIGANT_IN", -urine["VALUE"], urine["VALUE"]
    )
    logger.info(f"Loaded {len(urine)} urine output events across "
                f"{urine['ICUSTAY_ID'].nunique()} stays")
    return urine[["ICUSTAY_ID", "CHARTTIME", "VOLUME_ML"]]


def load_ventilation(mimic_dir, icustay_ids):
    """
    Load mechanical ventilation procedure intervals from PROCEDUREEVENTS_MV.

    Returns DataFrame with ICUSTAY_ID, STARTTIME, ENDTIME, VENT_TYPE
    where VENT_TYPE is 'INVASIVE' or 'NONINVASIVE'.
    """
    path = Path(mimic_dir)
    for ext in ["PROCEDUREEVENTS_MV.csv.gz", "PROCEDUREEVENTS_MV.csv"]:
        fpath = path / ext
        if fpath.exists():
            break
    else:
        logger.warning("PROCEDUREEVENTS_MV not found — skipping ventilation")
        return pd.DataFrame(columns=["ICUSTAY_ID", "STARTTIME", "ENDTIME", "VENT_TYPE"])

    logger.info(f"Loading ventilation from {fpath}")
    proc = pd.read_csv(
        fpath,
        usecols=["ICUSTAY_ID", "ITEMID", "STARTTIME", "ENDTIME"],
        dtype={"ICUSTAY_ID": "float64", "ITEMID": "int64"},
        low_memory=False,
    )
    proc = proc[proc["ITEMID"].isin(VENT_ITEMS.keys())]
    proc = proc[proc["ICUSTAY_ID"].isin(icustay_ids)]
    if len(proc) == 0:
        return pd.DataFrame(columns=["ICUSTAY_ID", "STARTTIME", "ENDTIME", "VENT_TYPE"])

    proc["STARTTIME"] = pd.to_datetime(proc["STARTTIME"], errors="coerce")
    proc["ENDTIME"] = pd.to_datetime(proc["ENDTIME"], errors="coerce")
    proc = proc.dropna(subset=["STARTTIME", "ENDTIME", "ICUSTAY_ID"])
    proc["ICUSTAY_ID"] = proc["ICUSTAY_ID"].astype(int)
    proc["VENT_TYPE"] = proc["ITEMID"].map(VENT_ITEMS).str.replace("_VENT", "", regex=False)
    logger.info(f"Loaded {len(proc)} ventilation intervals across "
                f"{proc['ICUSTAY_ID'].nunique()} stays")
    return proc[["ICUSTAY_ID", "STARTTIME", "ENDTIME", "VENT_TYPE"]]


def aggregate_treatments_hourly(vaso, urine, vent, icustays, max_hours=72):
    """
    Aggregate vasopressors, urine output, and ventilation onto the same
    hourly grid as build_hourly_timelines.

    Returns DataFrame with ICUSTAY_ID, HOUR, and TX_* columns.
    """
    icu = icustays[["ICUSTAY_ID", "INTIME"]].copy()
    icu["INTIME"] = pd.to_datetime(icu["INTIME"])
    icu["ICUSTAY_ID"] = icu["ICUSTAY_ID"].astype(int)

    # Build the (ICUSTAY_ID, HOUR) skeleton matching build_hourly_timelines bins
    hours_idx = []
    for stay_id, intime in zip(icu["ICUSTAY_ID"].values, icu["INTIME"].values):
        for h in range(max_hours):
            hours_idx.append((stay_id, h))
    skel = pd.DataFrame(hours_idx, columns=["ICUSTAY_ID", "HOUR"])
    skel = skel.merge(icu, on="ICUSTAY_ID")
    skel["BIN_START"] = skel["INTIME"] + pd.to_timedelta(skel["HOUR"], unit="h")
    skel["BIN_END"] = skel["BIN_START"] + pd.Timedelta(hours=1)

    # ---- Vasopressors: per-hour rate by drug ----
    drug_cols = {}
    for drug in ["NOREPI", "EPI", "DOPAMINE", "PHENYLEPH", "VASOPRESSIN"]:
        drug_cols[f"TX_{drug}_RATE"] = np.zeros(len(skel))

    if len(vaso) > 0:
        vaso = vaso.merge(icu, on="ICUSTAY_ID")
        # For each infusion, find which (stay, hour) bins it overlaps
        for _, row in vaso.iterrows():
            stay_id = row["ICUSTAY_ID"]
            start_h = max(0, int((row["STARTTIME"] - row["INTIME"]).total_seconds() // 3600))
            end_h = min(max_hours, int((row["ENDTIME"] - row["INTIME"]).total_seconds() // 3600) + 1)
            if end_h <= 0 or start_h >= max_hours:
                continue
            mask = (skel["ICUSTAY_ID"] == stay_id) & \
                   (skel["HOUR"] >= start_h) & (skel["HOUR"] < end_h)
            col = f"TX_{row['DRUG']}_RATE"
            if col in drug_cols and pd.notna(row["RATE_NORMALIZED"]):
                # If multiple infusions overlap, take the max rate
                idx = skel.index[mask]
                drug_cols[col][idx] = np.maximum(drug_cols[col][idx], row["RATE_NORMALIZED"])

    for col, vals in drug_cols.items():
        skel[col] = vals

    # Derived: any pressor active, count of distinct agents
    pressor_cols = [c for c in drug_cols.keys()]
    skel["TX_VASO_ANY"] = (skel[pressor_cols] > 0).any(axis=1).astype(int)
    skel["TX_VASO_N_AGENTS"] = (skel[pressor_cols] > 0).sum(axis=1).astype(int)

    # ---- Urine output: sum mL per hour ----
    if len(urine) > 0:
        urine = urine.merge(icu, on="ICUSTAY_ID")
        urine["HOUR"] = ((urine["CHARTTIME"] - urine["INTIME"]).dt.total_seconds() // 3600).astype("Int64")
        urine = urine[(urine["HOUR"] >= 0) & (urine["HOUR"] < max_hours)]
        urine_hourly = (
            urine.groupby(["ICUSTAY_ID", "HOUR"])["VOLUME_ML"]
            .sum()
            .reset_index()
            .rename(columns={"VOLUME_ML": "TX_URINE_ML"})
        )
        urine_hourly["HOUR"] = urine_hourly["HOUR"].astype(int)
        skel = skel.merge(urine_hourly, on=["ICUSTAY_ID", "HOUR"], how="left")
    else:
        skel["TX_URINE_ML"] = np.nan
    skel["TX_URINE_ML"] = skel["TX_URINE_ML"].fillna(0)

    # ---- Ventilation: boolean flags per hour ----
    skel["TX_VENT_INVASIVE"] = 0
    skel["TX_VENT_NONINVASIVE"] = 0
    if len(vent) > 0:
        vent = vent.merge(icu, on="ICUSTAY_ID")
        for _, row in vent.iterrows():
            stay_id = row["ICUSTAY_ID"]
            start_h = max(0, int((row["STARTTIME"] - row["INTIME"]).total_seconds() // 3600))
            end_h = min(max_hours, int((row["ENDTIME"] - row["INTIME"]).total_seconds() // 3600) + 1)
            if end_h <= 0 or start_h >= max_hours:
                continue
            mask = (skel["ICUSTAY_ID"] == stay_id) & \
                   (skel["HOUR"] >= start_h) & (skel["HOUR"] < end_h)
            col = f"TX_VENT_{row['VENT_TYPE']}"
            if col in skel.columns:
                skel.loc[mask, col] = 1

    tx_cols = [c for c in skel.columns if c.startswith("TX_")]
    return skel[["ICUSTAY_ID", "HOUR"] + tx_cols]


def compute_sofa_from_hourly(hourly):
    """
    Approximate SOFA score from available hourly features.

    Cardiovascular and renal components use treatment columns (TX_*) when
    present, falling back to MAP-only / creatinine-only when not. This is
    closer to the canonical Sepsis-3 SOFA than the v1 vitals-only version.
    """
    sofa = pd.DataFrame(index=hourly.index)
    sofa["ICUSTAY_ID"] = hourly["ICUSTAY_ID"]
    sofa["HOUR"] = hourly["HOUR"]
    sofa["SOFA"] = 0

    # Cardiovascular: take max of MAP-based and pressor-based scores
    cv_score = np.zeros(len(hourly), dtype=int)
    if "MAP" in hourly.columns:
        cv_score = np.maximum(cv_score, np.where(hourly["MAP"].fillna(99) < 70, 1, 0))
    # Pressor ladder (mcg/kg/min for catecholamines)
    if "TX_DOPAMINE_RATE" in hourly.columns:
        dop = hourly["TX_DOPAMINE_RATE"].fillna(0).values
        cv_score = np.maximum(cv_score, np.where(dop > 15, 4,
                                          np.where(dop > 5, 3,
                                          np.where(dop > 0, 2, 0))))
    if "TX_NOREPI_RATE" in hourly.columns:
        ne = hourly["TX_NOREPI_RATE"].fillna(0).values
        cv_score = np.maximum(cv_score, np.where(ne > 0.1, 4,
                                          np.where(ne > 0, 3, 0)))
    if "TX_EPI_RATE" in hourly.columns:
        ep = hourly["TX_EPI_RATE"].fillna(0).values
        cv_score = np.maximum(cv_score, np.where(ep > 0.1, 4,
                                          np.where(ep > 0, 3, 0)))
    sofa["SOFA"] += cv_score

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

    # Renal: max of creatinine-based and 24h-urine-output-based scores
    renal_score = np.zeros(len(hourly), dtype=int)
    if "CREATININE" in hourly.columns:
        cr = hourly["CREATININE"].fillna(0).values
        renal_score = np.maximum(renal_score, np.select(
            [cr >= 5, cr >= 3.5, cr >= 2, cr >= 1.2],
            [4, 3, 2, 1], default=0
        ))
    if "TX_URINE_ML" in hourly.columns:
        # 24-hour rolling sum per stay (ordered by HOUR)
        h_sorted = hourly.sort_values(["ICUSTAY_ID", "HOUR"])
        rolling = (
            h_sorted.groupby("ICUSTAY_ID")["TX_URINE_ML"]
            .rolling(window=24, min_periods=1).sum()
            .reset_index(level=0, drop=True)
        )
        # Re-align to original index order
        urine_24h = pd.Series(rolling.values, index=h_sorted.index).reindex(hourly.index).values
        renal_score = np.maximum(renal_score, np.where(urine_24h < 200, 4,
                                                np.where(urine_24h < 500, 3, 0)))
    sofa["SOFA"] += renal_score

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

    # 3b. Extract clinical treatments (vasopressors, urine output, ventilation)
    cohort_icu_ids = set(cohort["ICUSTAY_ID"].dropna().astype(int))
    vaso = load_vasopressors(mimic_dir, cohort_icu_ids)
    urine = load_urine_output(mimic_dir, cohort_icu_ids)
    vent = load_ventilation(mimic_dir, cohort_icu_ids)
    treatments = aggregate_treatments_hourly(
        vaso, urine, vent, cohort, max_hours=cfg["cohort"]["max_hours"]
    )
    logger.info(f"Treatments aggregated: {len(treatments)} (stay, hour) rows, "
                f"{(treatments['TX_VASO_ANY'] > 0).sum()} pressor-active rows")

    hourly = build_hourly_timelines(
        observations, cohort,
        max_hours=cfg["cohort"]["max_hours"],
        bin_minutes=cfg["cohort"]["bin_window_minutes"],
        treatments=treatments,
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
