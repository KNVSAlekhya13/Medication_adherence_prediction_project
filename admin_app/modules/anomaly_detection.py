import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(adherence, contamination="auto"):
    df=adherence.copy()
    if df.empty:
        df["anomaly"]=False; df["anomaly_score"]=0.0; return df
    if "adherence_rate" not in df.columns:
        df["adherence_rate"]=(df["taken_doses"]/df["scheduled_doses"].replace(0,pd.NA)).fillna(0).clip(0,1)
    df["late_rate"]=(df["late_doses"]/df["taken_doses"].replace(0,float("nan"))).fillna(0).clip(0,1)
    cols=["scheduled_doses","adherence_rate","late_rate","taken_doses"]
    if len(df)<10:
        df["anomaly"]=False; df["anomaly_score"]=0.0; return df
    model=IsolationForest(contamination=contamination,random_state=42,n_estimators=300)
    df["anomaly"]=model.fit_predict(df[cols].fillna(0))==-1
    df["anomaly_score"]=model.decision_function(df[cols].fillna(0)).round(4)
    return df
