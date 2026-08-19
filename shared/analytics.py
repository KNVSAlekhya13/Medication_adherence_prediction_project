"""Patient-level adherence feature generation for live app predictions."""
from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
import pandas as pd
from .db import expected_slots_for_date


def patient_adherence_history(conn, patient_id, cutoff_date=None):
    """Build completed-day adherence from scheduled slots, not merely logged doses."""
    if cutoff_date is None:
        cutoff_date = date.today() - timedelta(days=1)
    elif isinstance(cutoff_date, str):
        cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    first = conn.execute("SELECT MIN(date(created_at)) FROM medications WHERE patient_id=? AND active=1", (patient_id,)).fetchone()[0]
    if not first:
        return pd.DataFrame(columns=["patient_id","date","scheduled_doses","taken_doses","late_doses","adherence_rate"])
    start = datetime.strptime(first, "%Y-%m-%d").date()
    rows = []
    day = start
    while day <= cutoff_date:
        slots = expected_slots_for_date(conn, patient_id, day)
        if slots:
            user = conn.execute("SELECT user_id FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
            user_id = user["user_id"] if user else None
            taken = late = 0
            if user_id:
                for med, slot in slots:
                    d = conn.execute(
                        "SELECT status,taken_at FROM doses WHERE user_id=? AND medicine_id=? AND dose_date=? AND scheduled_time=?",
                        (user_id, med["id"], day.isoformat(), slot)
                    ).fetchone()
                    if d and d["status"] == "taken":
                        taken += 1
                        if d["taken_at"]:
                            try:
                                t = datetime.fromisoformat(d["taken_at"])
                                s = datetime.fromisoformat(f"{day.isoformat()}T{slot}:00")
                                if t.tzinfo is not None:
                                    t = t.replace(tzinfo=None)
                                if s.tzinfo is not None:
                                    s = s.replace(tzinfo=None)
                                late += int(t > s + timedelta(minutes=30))
                            except ValueError:
                                pass
            rows.append({"patient_id": patient_id, "date": pd.Timestamp(day),
                         "scheduled_doses": len(slots), "taken_doses": taken,
                         "late_doses": late, "adherence_rate": taken / len(slots) if slots else 0.0})
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def build_live_patient_features(conn, patient_id, cutoff_date=None):
    if cutoff_date is None:
        cutoff_date = date.today() - timedelta(days=1)
    history = patient_adherence_history(conn, patient_id, cutoff_date)
    patient = conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
    if not patient:
        return None
    if history.empty:
        return {
            "patient_id": patient_id, "age": patient["age"] or 0,
            "gender": patient["gender"] or "", "condition": patient["condition"] or "",
            "reminder_enabled": patient["reminder_enabled"], "mean_adherence": 0.0,
            "hist_adherence": 0.0, "last_adherence": 0.0, "total_scheduled": 0,
            "total_taken": 0, "late_doses": 0, "days_recorded": 0, "history_days": 0,
            "missed_doses": 0, "miss_rate": 0.0, "recent_missed": 0,
        }
    recent = history.tail(min(7, len(history)))
    scheduled = history.scheduled_doses.sum(); taken = history.taken_doses.sum()
    recent_s = recent.scheduled_doses.sum(); recent_t = recent.taken_doses.sum()
    mean = taken / scheduled if scheduled else 0.0
    return {
        "patient_id": patient_id, "age": patient["age"] or 0,
        "gender": patient["gender"] or "", "condition": patient["condition"] or "",
        "reminder_enabled": patient["reminder_enabled"], "mean_adherence": mean,
        "hist_adherence": mean, "last_adherence": recent_t / recent_s if recent_s else 0.0,
        "total_scheduled": int(scheduled), "total_taken": int(taken),
        "late_doses": int(history.late_doses.sum()), "days_recorded": int(history.date.nunique()),
        "history_days": int(len(history)), "missed_doses": int(max(scheduled - taken, 0)),
        "miss_rate": max(1.0 - mean, 0.0),
        "recent_missed": int(max(recent_s - recent_t, 0)),
    }
