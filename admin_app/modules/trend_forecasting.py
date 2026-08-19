"""Adherence forecasting with transparent baselines and rolling backtesting.

The system compares a linear trend with a recent-mean baseline and reports
backtest error. Forecasts are exploratory research outputs, not clinical predictions.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def _series(adherence, patient_id):
    df=adherence[adherence["patient_id"].astype(str)==str(patient_id)].copy().sort_values("date")
    if "adherence_rate" not in df.columns:
        df["adherence_rate"]=(df.taken_doses/df.scheduled_doses.replace(0,pd.NA)).fillna(0).clip(0,1)
    return df

def _predict(train, periods, method):
    y=train.adherence_rate.astype(float).to_numpy()
    if method=="recent_mean":
        value=float(np.mean(y[-7:])); return np.repeat(value, periods)
    X=np.arange(len(y)).reshape(-1,1); model=LinearRegression().fit(X,y)
    return model.predict(np.arange(len(y),len(y)+periods).reshape(-1,1)).clip(0,1)

def _select_method(df, min_train=7):
    if len(df)<min_train+3: return "recent_mean", None
    errors={}
    for method in ("recent_mean","linear_trend"):
        actual=[]; pred=[]
        for i in range(min_train,len(df)):
            train=df.iloc[:i]
            pred.append(float(_predict(train,1,method)[0])); actual.append(float(df.iloc[i].adherence_rate))
        errors[method]={"mae":mean_absolute_error(actual,pred),"rmse":mean_squared_error(actual,pred)**0.5,"n_test":len(actual)}
    best=min(errors,key=lambda m:(errors[m]["mae"],errors[m]["rmse"]))
    return best, errors

def forecast_adherence(adherence, patient_id, periods=7):
    df=_series(adherence,patient_id)
    if len(df)<7:return pd.DataFrame()
    method,errors=_select_method(df)
    pred=_predict(df,periods,method)
    residual=df.adherence_rate.astype(float).to_numpy()-_predict(df,len(df),"linear_trend")
    se=float(np.std(residual,ddof=1)) if len(residual)>2 else 0.1
    dates=pd.date_range(pd.to_datetime(df.date.max())+pd.Timedelta(days=1),periods=periods)
    out=pd.DataFrame({"date":dates,"forecast_adherence":pred,"lower_bound":np.clip(pred-1.96*se,0,1),"upper_bound":np.clip(pred+1.96*se,0,1)})
    out.attrs["method"]=method; out.attrs["backtest"]=errors
    return out

def backtest_forecast(adherence,patient_id,min_train=7):
    df=_series(adherence,patient_id)
    if len(df)<min_train+3:return {"mae":None,"rmse":None,"n_test":0,"method":"Insufficient data"}
    method,errors=_select_method(df,min_train)
    return {**errors[method],"method":method}

def trend_direction(adherence,patient_id):
    f=forecast_adherence(adherence,patient_id,3)
    if f.empty:return "Insufficient data"
    delta=float(f.forecast_adherence.iloc[-1]-f.forecast_adherence.iloc[0])
    return "Improving" if delta>.03 else "Declining" if delta<-.03 else "Stable"
