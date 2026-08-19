import pandas as pd

# Research/demo thresholds. They are configurable and should be justified/re-tuned on real study data.
DEFAULT_THRESHOLDS={"missed_events":2,"late_events":2,"decline_points":0.15,"decline_window_days":3}

def detect_patterns(adherence, thresholds=None):
    cfg={**DEFAULT_THRESHOLDS,**(thresholds or {})}; results=[]
    for patient_id,group in adherence.groupby("patient_id"):
        group=group.sort_values("date"); missed=group["taken_doses"]<group["scheduled_doses"]
        if int(missed.sum())>=cfg["missed_events"]: results.append({"patient_id":patient_id,"pattern":f"Repeated missed dose days (≥{cfg['missed_events']})","count":int(missed.sum())})
        if int(group["late_doses"].sum())>=cfg["late_events"]: results.append({"patient_id":patient_id,"pattern":f"Repeated late intake (≥{cfg['late_events']} events)","count":int(group["late_doses"].sum())})
        if len(group)>=6:
            recent=group.tail(3)["adherence_rate"].mean(); earlier=group.head(3)["adherence_rate"].mean()
            if recent < earlier-cfg["decline_points"]: results.append({"patient_id":patient_id,"pattern":f"Recent adherence decline (>{cfg['decline_points']:.0%})","count":1})
    return pd.DataFrame(results,columns=["patient_id","pattern","count"])
