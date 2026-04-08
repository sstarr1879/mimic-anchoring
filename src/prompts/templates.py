"""
Phase 2: Prompt templates for presenting patient timelines to LLaMA.

Handles formatting hourly vitals/labs into structured text prompts,
and defines the expected output format for risk assessments.
"""

SYSTEM_PROMPT = """You are an ICU patient monitoring system. Your role is to assess sepsis risk based on sequential vital signs and laboratory values.

At each observation, you must provide:
1. A sepsis risk probability between 0.0 and 1.0
2. A brief clinical reasoning explaining your assessment

Format your response exactly as:
RISK: <probability>
REASONING: <one or two sentences explaining your assessment>"""


HOUR_TEMPLATE = """Hour {hour} vitals/labs:
{observations}"""


INCREMENTAL_USER_PROMPT = """Based on the patient data so far, assess the current sepsis risk.

{timeline}

Provide your assessment for the most recent hour."""


REASSESS_PROMPT = """Ignore all previous context. You are seeing this patient for the first time. Based ONLY on the following current vitals and recent trends, assess sepsis risk from scratch.

{timeline}

Provide your assessment."""


FEATURE_DISPLAY_NAMES = {
    "HR": "Heart Rate",
    "MAP": "Mean Arterial Pressure",
    "TEMP_F": "Temperature (F)",
    "TEMP_C": "Temperature (C)",
    "SPO2": "SpO2",
    "RR": "Respiratory Rate",
    "GCS": "Glasgow Coma Scale",
    "PO2": "PaO2",
    "FIO2": "FiO2",
    "PLATELETS": "Platelets",
    "BILIRUBIN": "Bilirubin",
    "CREATININE": "Creatinine",
    "WBC": "White Blood Cells",
    "LACTATE": "Lactate",
    # Clinical treatments (TX_*) — rendered sparsely (only on change/active)
    "TX_NOREPI_RATE": "Norepinephrine (mcg/kg/min)",
    "TX_EPI_RATE": "Epinephrine (mcg/kg/min)",
    "TX_DOPAMINE_RATE": "Dopamine (mcg/kg/min)",
    "TX_PHENYLEPH_RATE": "Phenylephrine",
    "TX_VASOPRESSIN_RATE": "Vasopressin (units/hr)",
    "TX_VASO_N_AGENTS": "Vasopressors active (n)",
    "TX_URINE_ML": "Urine Output (mL)",
    "TX_VENT_INVASIVE": "Mechanical Ventilation",
    "TX_VENT_NONINVASIVE": "Non-invasive Ventilation",
}

# Treatment columns are rendered sparsely (only on change events) under
# chronological/reverse ordering. Listed here so format_hour_observations
# can decide whether a TX value should be emitted on a given hour.
TX_FEATURE_KEYS = {k for k in FEATURE_DISPLAY_NAMES if k.startswith("TX_")}


def format_hour_observations(row, features=None, prev_row=None, sparse_tx=True):
    """
    Format a single hour's observations as readable text.

    Treatment columns (TX_*) are rendered sparsely by default: a TX value
    only appears when (a) it's active and nonzero AND (b) it differs from
    the previous hour's value. This keeps prompts compact while making
    treatment *changes* (start, escalate, wean, stop) maximally salient
    — the signal that anchoring detection actually needs.

    Args:
        row: pandas Series for the current hour's observations.
        features: optional explicit feature list. Defaults to all known.
        prev_row: optional previous hour's row, for sparse TX rendering.
        sparse_tx: if False, render every TX value present (used by
            shuffled ordering where "previous" has no temporal meaning,
            and by single-hour prompts).
    """
    if features is None:
        features = [c for c in row.index if c in FEATURE_DISPLAY_NAMES]

    lines = []
    for feat in features:
        if feat not in row.index or pd.isna(row[feat]):
            continue
        val = row[feat]
        is_tx = feat in TX_FEATURE_KEYS

        if is_tx and sparse_tx:
            prev_val = prev_row[feat] if (prev_row is not None and feat in prev_row.index) else 0
            if pd.isna(prev_val):
                prev_val = 0
            # Sparse rule: render only if value is nonzero AND changed,
            # OR if value is zero but previous was nonzero (a stop event).
            if val == 0 and prev_val == 0:
                continue
            if val == prev_val and prev_row is not None:
                continue

        display = FEATURE_DISPLAY_NAMES.get(feat, feat)
        # Boolean-like vent flags render as on/off
        if feat in ("TX_VENT_INVASIVE", "TX_VENT_NONINVASIVE"):
            lines.append(f"  {display}: {'ON' if val else 'OFF'}")
        elif feat == "TX_VASO_N_AGENTS":
            lines.append(f"  {display}: {int(val)}")
        else:
            lines.append(f"  {display}: {val:.2f}" if is_tx else f"  {display}: {val:.1f}")

    return "\n".join(lines) if lines else "  No new observations"


def build_timeline_prompt(patient_hours, up_to_hour=None, ordering="chronological"):
    """
    Build a full timeline prompt from a patient's hourly data.

    Args:
        patient_hours: DataFrame of one patient's hourly observations.
        up_to_hour: Include hours up to this value (inclusive). None = all hours.
        ordering: "chronological", "reverse", or "shuffled"

    Returns:
        Formatted timeline string.
    """
    import pandas as pd

    df = patient_hours.sort_values("HOUR").copy()
    if up_to_hour is not None:
        df = df[df["HOUR"] <= up_to_hour]

    # Sparse TX rendering needs a meaningful "previous hour" — only valid
    # when rows are in temporal order. Under shuffled ordering, fall back
    # to dense rendering so the model still sees treatment values.
    sparse_tx = ordering != "shuffled"

    if ordering == "reverse":
        df = df.iloc[::-1].copy()
    elif ordering == "shuffled":
        df = df.sample(frac=1, random_state=42).copy()

    blocks = []
    prev = None
    for _, row in df.iterrows():
        obs_text = format_hour_observations(row, prev_row=prev, sparse_tx=sparse_tx)
        blocks.append(HOUR_TEMPLATE.format(hour=int(row["HOUR"]), observations=obs_text))
        prev = row

    return "\n\n".join(blocks)


def build_incremental_prompts(patient_hours, ordering="chronological"):
    """
    Build a sequence of prompts, one per hour, for incremental assessment.

    This is used for belief update elasticity: we send the model
    the timeline up to hour 0, then up to hour 1, etc., collecting
    a risk assessment at each step.

    Returns:
        List of (hour, system_prompt, user_prompt) tuples.
    """
    import pandas as pd

    df = patient_hours.sort_values("HOUR")
    hours = sorted(df["HOUR"].unique())

    prompts = []
    for h in hours:
        timeline = build_timeline_prompt(patient_hours, up_to_hour=h, ordering=ordering)
        user_prompt = INCREMENTAL_USER_PROMPT.format(timeline=timeline)
        prompts.append((h, SYSTEM_PROMPT, user_prompt))

    return prompts


def build_multiturn_messages(patient_hours, ordering="chronological"):
    """
    Build a multi-turn conversation for trajectory-aware inference.

    Instead of stuffing the full timeline into one prompt, each hour
    becomes a separate user turn, and the model responds at each step.
    This matches the trajectory-aware SFT training format.

    Args:
        patient_hours: DataFrame of one patient's hourly observations.
        ordering: "chronological", "reverse", or "shuffled"

    Returns:
        List of (hour, messages_so_far) tuples.
        Each messages_so_far is a list of {role, content} dicts
        ready to send to the model, with prior assistant responses
        to be filled in during inference.
    """
    df = patient_hours.sort_values("HOUR").copy()
    sparse_tx = ordering != "shuffled"

    if ordering == "reverse":
        df = df.iloc[::-1].copy()
    elif ordering == "shuffled":
        df = df.sample(frac=1, random_state=42).copy()

    turns = []
    prev = None
    for _, row in df.iterrows():
        obs_text = format_hour_observations(row, prev_row=prev, sparse_tx=sparse_tx)
        user_msg = (
            f"Hour {int(row['HOUR'])} vitals/labs:\n{obs_text}\n\n"
            f"Update your sepsis risk assessment."
        )
        turns.append((int(row["HOUR"]), user_msg))
        prev = row

    return turns


# Need pandas for format_hour_observations
import pandas as pd
