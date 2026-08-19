def simulate_improvement(row, improvement=0.10, predictor=None):
    current=float(row["mean_adherence"]); projected=min(current+max(float(improvement),0.0),1.0)
    result={"current_adherence":round(current,3),"projected_adherence":round(projected,3),"improvement":round(projected-current,3),"interpretation":"Scenario only; it does not estimate a causal treatment effect."}
    if predictor:
        base=predictor(row.to_dict() if hasattr(row,"to_dict") else dict(row))
        simulated=dict(row.to_dict() if hasattr(row,"to_dict") else row)
        simulated["mean_adherence"]=projected; simulated["hist_adherence"]=projected; simulated["last_adherence"]=projected; simulated["miss_rate"]=1-projected
        simulated["missed_doses"]=round(float(simulated.get("total_scheduled",0))*(1-projected))
        after=predictor(simulated)
        result.update({"current_risk":base.get("risk_probability"),"projected_risk":after.get("risk_probability"),"model_recomputed":after.get("prediction_available",False)})
    return result
