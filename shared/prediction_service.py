"""Single prediction service shared by patient and admin applications."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "admin_app" / "models"
MODEL = joblib.load(MODEL_DIR / "adherence_model.pkl")
FEATURES = joblib.load(MODEL_DIR / "feature_columns.pkl")
CONFIG = joblib.load(MODEL_DIR / "prediction_config.pkl")
THRESHOLD = float(CONFIG["prediction_threshold"])
RISK_DEFINITION = "Probability that future adherence will be below the study target (70%)."
FEATURE_SOURCE_MAP = {
    "age": "age", "reminder": "reminder_enabled", "hist": "hist_adherence",
    "recent": "last_adherence", "missed": "missed_doses", "late": "late_doses",
    "miss_rate": "miss_rate", "recent_missed": "recent_missed",
}
DEFAULTS = {"age": 0, "reminder": 0, "hist": 0.0, "recent": 0.0, "missed": 0,
            "late": 0, "miss_rate": 0.0, "recent_missed": 0}


def _input(features_df):
    df = pd.DataFrame(features_df).copy() if not isinstance(features_df, pd.DataFrame) else features_df.copy()
    model_input = pd.DataFrame(index=df.index)
    for f in FEATURES:
        src = FEATURE_SOURCE_MAP.get(f, f)
        model_input[f] = df[src] if src in df.columns else DEFAULTS.get(f, 0)
    return model_input.apply(pd.to_numeric, errors="coerce").fillna(0)[FEATURES]


def predict_risk(patient_features):
    df = pd.DataFrame([patient_features]) if isinstance(patient_features, dict) else pd.DataFrame(patient_features)
    history_days = int(pd.to_numeric(df.iloc[0].get("history_days", 0), errors="coerce") or 0)
    min_days = int(CONFIG.get("minimum_history_days", 7))
    if history_days < min_days:
        return {"risk_label": "Insufficient Data", "risk_probability": 0.0,
                "good_adherence_probability": 0.0, "prediction_available": False,
                "message": f"At least {min_days} completed adherence days are required before estimating risk."}
    prob = float(MODEL.predict_proba(_input(df))[0][1])
    return {"risk_label": "Low Risk" if prob >= THRESHOLD else "High Risk",
            "risk_probability": round(1 - prob, 3), "good_adherence_probability": round(prob, 3),
            "prediction_available": True,
            "message": "Research estimate only: probability of future adherence below 70%, using only data before the cutoff; not a diagnosis.", "risk_definition": RISK_DEFINITION}


def predict_risk_batch(features_df):
    df = features_df.copy()
    probs = MODEL.predict_proba(_input(df))[:, 1]
    history = pd.to_numeric(df.get("history_days", 0), errors="coerce").fillna(0)
    enough = history >= int(CONFIG.get("minimum_history_days", 7))
    labels = pd.Series("Insufficient Data", index=df.index)
    labels.loc[enough] = pd.Series((probs[enough] >= THRESHOLD), index=df.index[enough]).map({True: "Low Risk", False: "High Risk"})
    risk = pd.Series((1 - probs).round(3), index=df.index); good = pd.Series(probs.round(3), index=df.index)
    risk.loc[~enough] = 0.0; good.loc[~enough] = 0.0
    return pd.DataFrame({"risk_label": labels, "risk_probability": risk,
                         "good_adherence_probability": good, "prediction_available": enough}, index=df.index)
