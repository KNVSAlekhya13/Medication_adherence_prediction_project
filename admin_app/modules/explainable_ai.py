"""Local model explanation using feature ablation/sensitivity."""
import sys
from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from shared.prediction_service import MODEL, FEATURES, FEATURE_SOURCE_MAP

LABELS = {"hist":"historical adherence","recent":"recent adherence","missed":"missed doses","late":"late doses",
          "miss_rate":"miss rate","recent_missed":"recent missed doses","age":"age","reminder":"reminder setting"}


def explain_prediction(row, risk_result):
    if not risk_result.get("prediction_available"):
        return {"risk":risk_result["risk_label"],"probability":0.0,
                "explanation":"A local explanation is not available until sufficient completed adherence history exists.","feature_importance":[]}
    base = pd.DataFrame([row]); from shared.prediction_service import _input
    base_prob = float(MODEL.predict_proba(_input(base))[0][1])
    values=[]
    medians={"age":40,"reminder":1,"hist":0.70,"recent":0.70,"missed":0,"late":0,"miss_rate":0.30,"recent_missed":0}
    for f in FEATURES:
        changed=base.copy(); src=FEATURE_SOURCE_MAP.get(f,f); changed[src]=medians.get(f,0)
        p=float(MODEL.predict_proba(_input(changed))[0][1])
        delta=base_prob-p
        values.append((abs(delta),f,delta))
    values.sort(reverse=True)
    details=[]
    for _,f,delta in values[:3]:
        if delta>0: details.append(f"{LABELS.get(f,f)} supports good-adherence probability")
        elif delta<0: details.append(f"{LABELS.get(f,f)} lowers good-adherence probability")
        else: details.append(f"{LABELS.get(f,f)} has little local effect")
    return {"risk":risk_result["risk_label"],"probability":risk_result["risk_probability"],
            "explanation":"Local model sensitivity: " + "; ".join(details) + ". This is a model explanation, not a causal medical conclusion.",
            "feature_importance":[(LABELS.get(f,f),round(delta,4)) for _,f,delta in values]}
