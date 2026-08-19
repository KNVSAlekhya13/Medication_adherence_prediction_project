"""Rule-based adherence factor screening.

This module is deliberately NOT called statistical factor analysis: it does
not perform latent-variable factor analysis, PCA, or inferential statistics.
Scores are transparent descriptive indicators for research exploration.
"""
def analyze_factors(row):
    factors=[]
    if float(row.get("mean_adherence",0)) < 0.80: factors.append("Low overall adherence")
    if float(row.get("miss_rate",0)) > 0.20: factors.append("Frequent missed doses")
    if float(row.get("late_doses",0)) > 0: factors.append("Late medication intake")
    if int(row.get("days_recorded",0)) < 7: factors.append("Limited adherence history")
    return factors or ["No threshold-based factor detected"]

def rank_factors(row):
    factors={
        "Low adherence indicator": max(0,1-float(row.get("mean_adherence",0))),
        "Missed-dose indicator": float(row.get("miss_rate",0)),
        "Late-dose indicator": min(float(row.get("late_doses",0))/max(float(row.get("days_recorded",0)),1),1)
    }
    return sorted(factors.items(),key=lambda x:x[1],reverse=True)
