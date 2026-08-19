"""Transparent, rule-based adherence support suggestions.

These are NOT model predictions and are NOT clinical treatment recommendations.
Thresholds are protocol parameters that should be reviewed by the research team.
"""
DEFAULT_RULES={"high_adherence_cutoff":0.80,"low_adherence_cutoff":0.60,"miss_rate_cutoff":0.20}
def recommend_intervention(row, rules=None):
    cfg={**DEFAULT_RULES,**(rules or {})}; recommendations=[]
    adherence=float(row.get("mean_adherence",0)); miss_rate=float(row.get("miss_rate",0)); late=int(row.get("late_doses",0))
    if adherence < cfg["low_adherence_cutoff"]:
        recommendations.append("Review barriers to medication-taking and consider appropriate follow-up.")
    elif adherence < cfg["high_adherence_cutoff"]:
        recommendations.append("Consider stronger reminders and review the daily medication routine.")
    else:
        recommendations.append("Continue the current adherence-support routine.")
    if miss_rate > cfg["miss_rate_cutoff"]:
        recommendations.append("Explore reasons for repeated missed doses with the participant.")
    if late > 0:
        recommendations.append("Review medication timing and reminder schedule.")
    return recommendations
def intervention_level(row, rules=None):
    cfg={**DEFAULT_RULES,**(rules or {})}; a=float(row.get("mean_adherence",0))
    return "High Priority" if a<cfg["low_adherence_cutoff"] else "Medium Priority" if a<cfg["high_adherence_cutoff"] else "Low Priority"
