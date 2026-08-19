"""Validated, database-backed data access and temporal feature engineering."""
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.db import get_connection

PATIENT_COLUMNS = ["patient_id", "age", "gender", "condition", "reminder_enabled"]
ADHERENCE_COLUMNS = ["patient_id", "date", "scheduled_doses", "taken_doses", "late_doses"]


def validate_patients(patients):
    df = patients.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in PATIENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing patient column(s): {', '.join(missing)}")
    df = df[PATIENT_COLUMNS].copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    mapping = {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0, 1: 1, 0: 0}
    df["reminder_enabled"] = df["reminder_enabled"].map(mapping)
    invalid = (
        df["patient_id"].eq("") | df["age"].isna() | ~df["age"].between(0, 120)
        | df["reminder_enabled"].isna()
    )
    if invalid.any():
        raise ValueError(f"Invalid patient rows: {int(invalid.sum())}. Check patient_id, age (0-120), and reminder_enabled.")
    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient_id values are not allowed in an upload.")
    return df


def clean_adherence_data(adherence):
    df = adherence.copy()
    missing = [c for c in ADHERENCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Adherence data is missing columns: {', '.join(missing)}")
    df = df[ADHERENCE_COLUMNS].copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["scheduled_doses", "taken_doses", "late_doses"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    invalid = (
        df["patient_id"].eq("") | df["date"].isna()
        | df[["scheduled_doses", "taken_doses", "late_doses"]].isna().any(axis=1)
        | (df[["scheduled_doses", "taken_doses", "late_doses"]] < 0).any(axis=1)
        | (df["taken_doses"] > df["scheduled_doses"])
        | (df["late_doses"] > df["taken_doses"])
    )
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid adherence record(s). Data was not silently corrected.")
    zero = df["scheduled_doses"].eq(0)
    df["adherence_rate"] = 0.0
    df.loc[~zero, "adherence_rate"] = (df.loc[~zero, "taken_doses"] / df.loc[~zero, "scheduled_doses"]).clip(0, 1)
    return df.sort_values(["patient_id", "date"])


def load_data(data_dir=None):
    """Load research/demo data directly from the research dataset.

    This intentionally does NOT import research records into the operational
    patient database. The bundled dataset is synthetic and must remain isolated.
    """
    data_dir = Path(data_dir or (ROOT / "admin_app" / "data"))
    required = [data_dir/"patients.csv", data_dir/"medications.csv", data_dir/"adherence.csv"]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Research dataset is incomplete: " + ", ".join(missing))
    patients = validate_patients(pd.read_csv(required[0]))
    medications = pd.read_csv(required[1])
    adherence = clean_adherence_data(pd.read_csv(required[2]))
    unknown = set(adherence.patient_id) - set(patients.patient_id)
    if unknown:
        raise ValueError(f"Research adherence records reference unknown patients: {', '.join(sorted(unknown))}")
    return patients, medications, adherence


def build_patient_features(adherence, patients=None, cutoff_date=None):
    """Build features only from records on/before cutoff_date to prevent temporal leakage."""
    df = clean_adherence_data(adherence)
    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        df = df[df["date"] <= cutoff]
    rows = []
    for patient_id, group in df.groupby("patient_id"):
        group = group.sort_values("date")
        recent = group.tail(min(7, len(group)))
        scheduled = group["scheduled_doses"].sum()
        taken = group["taken_doses"].sum()
        mean_rate = taken / scheduled if scheduled else 0.0
        recent_sched = recent["scheduled_doses"].sum()
        recent_rate = recent["taken_doses"].sum() / recent_sched if recent_sched else 0.0
        rows.append({
            "patient_id": patient_id,
            "mean_adherence": mean_rate,
            "hist_adherence": mean_rate,
            "last_adherence": recent_rate,
            "total_scheduled": int(scheduled),
            "total_taken": int(taken),
            "late_doses": int(group["late_doses"].sum()),
            "days_recorded": int(group["date"].nunique()),
            "history_days": int(len(group)),
            "missed_doses": int(max(scheduled - taken, 0)),
            "miss_rate": max(1.0 - mean_rate, 0.0),
            "recent_missed": int(max(recent_sched - recent["taken_doses"].sum(), 0)),
        })
    features = pd.DataFrame(rows)
    if patients is not None:
        features = patients[PATIENT_COLUMNS].merge(features, on="patient_id", how="left")
        for c in ["mean_adherence", "hist_adherence", "last_adherence", "miss_rate"]:
            features[c] = features[c].fillna(0.0)
        for c in ["total_scheduled", "total_taken", "late_doses", "days_recorded", "history_days", "missed_doses", "recent_missed"]:
            features[c] = features[c].fillna(0).astype(int)
    return features


def get_patient_history(adherence, patient_id, cutoff_date=None):
    df = clean_adherence_data(adherence)
    df = df[df.patient_id.astype(str) == str(patient_id)]
    if cutoff_date is not None:
        df = df[df.date <= pd.Timestamp(cutoff_date)]
    return df.sort_values("date")


def get_adherence_summary(adherence):
    df = clean_adherence_data(adherence)
    scheduled = int(df.scheduled_doses.sum()); taken = int(df.taken_doses.sum())
    return {"total_scheduled": scheduled, "total_taken": taken, "total_missed": scheduled - taken,
            "overall_adherence": taken / scheduled if scheduled else 0.0}
