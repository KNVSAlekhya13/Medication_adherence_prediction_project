"""Shared MediTrack database used by patient and admin applications."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "meditrack.db"
CSV_DIR = ROOT / "admin_app" / "data"


def get_connection(db_path: Path = DB_PATH):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _column_names(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _schedule_times(frequency: str, time_text: str) -> list[str]:
    """Normalize explicit medication times. Explicit times are authoritative."""
    supplied = [x.strip() for x in str(time_text or "").split(",") if x.strip()]
    valid = []
    for value in supplied:
        try:
            datetime.strptime(value, "%H:%M")
            if value not in valid:
                valid.append(value)
        except ValueError:
            pass
    freq = str(frequency or "Once daily").strip().lower()
    if "four" in freq or "4" in freq: count, interval = 4, 6
    elif "three" in freq or "3" in freq: count, interval = 3, 8
    elif "twice" in freq or "two" in freq or "2" in freq: count, interval = 2, 12
    elif "every" in freq and "hour" in freq:
        m = re.search(r"(\d+)\s*hour", freq)
        count, interval = (max(1, 24 // int(m.group(1))), int(m.group(1))) if m else (1, 24)
    else: count, interval = 1, 24
    if len(valid) >= count:
        return sorted(valid[:count])
    base = datetime.strptime(valid[0] if valid else "08:00", "%H:%M")
    for i in range(len(valid), count):
        candidate=(base + timedelta(hours=interval*i)).strftime("%H:%M")
        if candidate not in valid: valid.append(candidate)
    return sorted(valid[:count])


def schedule_times(frequency: str, time_text: str) -> list[str]:
    return _schedule_times(frequency, time_text)


def initialize_database(import_csv: bool = True):
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            initials TEXT NOT NULL,
            age INTEGER,
            gender TEXT DEFAULT '',
            condition TEXT DEFAULT '',
            timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',
            reminder_enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS patients(
            patient_id TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            condition TEXT,
            timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',
            reminder_enabled INTEGER NOT NULL DEFAULT 1,
            user_id INTEGER UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS medications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medication_id TEXT UNIQUE,
            patient_id TEXT NOT NULL,
            medication_name TEXT NOT NULL,
            dosage TEXT NOT NULL,
            frequency TEXT NOT NULL,
            time TEXT NOT NULL DEFAULT '08:00',
            instructions TEXT DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS doses(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            medicine_id INTEGER NOT NULL,
            dose_date TEXT NOT NULL,
            scheduled_time TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('scheduled','taken','missed','snoozed')),
            taken_at TEXT,
            snoozed_until TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, medicine_id, dose_date, scheduled_time),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(medicine_id) REFERENCES medications(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS adherence(
            patient_id TEXT NOT NULL,
            date TEXT NOT NULL,
            scheduled_doses INTEGER NOT NULL DEFAULT 0,
            taken_doses INTEGER NOT NULL DEFAULT 0,
            late_doses INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(patient_id, date),
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS caregiver_accounts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS caregiver_links(
            caregiver_id INTEGER NOT NULL,
            patient_id TEXT NOT NULL,
            approved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(caregiver_id, patient_id),
            FOREIGN KEY(caregiver_id) REFERENCES caregiver_accounts(id) ON DELETE CASCADE,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS caregiver_requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caregiver_id INTEGER NOT NULL,
            patient_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','approved','revoked','rejected')),
            requested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            decided_at TEXT,
            UNIQUE(caregiver_id, patient_id),
            FOREIGN KEY(caregiver_id) REFERENCES caregiver_accounts(id) ON DELETE CASCADE,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS audit_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_type TEXT NOT NULL,
            actor_id TEXT,
            action TEXT NOT NULL,
            entity_type TEXT,
            entity_id TEXT,
            details TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    # Lightweight migration for databases created by older MediTrack builds.
    if "snoozed_until" not in _column_names(conn, "doses"):
        conn.execute("ALTER TABLE doses ADD COLUMN snoozed_until TEXT")

    # Safe migrations for databases created by an earlier fixed build.
    user_cols = _column_names(conn, "users")
    for col, definition in {
        "patient_id": "TEXT", "age": "INTEGER", "gender": "TEXT DEFAULT ''",
        "condition": "TEXT DEFAULT ''", "timezone": "TEXT NOT NULL DEFAULT 'Asia/Kolkata'",
        "reminder_enabled": "INTEGER NOT NULL DEFAULT 1",
        "created_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
    }.items():
        if col not in user_cols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
    patient_cols = _column_names(conn, "patients")
    if "timezone" not in patient_cols:
        conn.execute("ALTER TABLE patients ADD COLUMN timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata'")
    link_cols = _column_names(conn, "caregiver_links")
    if "approved_at" not in link_cols:
        conn.execute("ALTER TABLE caregiver_links ADD COLUMN approved_at TEXT NOT NULL DEFAULT ''")
    conn.commit()
    if import_csv:
        _import_seed_csv_if_empty(conn)
    conn.commit()
    conn.close()


def _import_seed_csv_if_empty(conn):
    patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if patient_count:
        return
    p = CSV_DIR / "patients.csv"
    m = CSV_DIR / "medications.csv"
    a = CSV_DIR / "adherence.csv"
    if not (p.exists() and m.exists() and a.exists()):
        return
    patients = pd.read_csv(p)
    meds = pd.read_csv(m)
    adherence = pd.read_csv(a)
    for _, r in patients.iterrows():
        conn.execute(
            "INSERT OR IGNORE INTO patients(patient_id,age,gender,condition,reminder_enabled) VALUES(?,?,?,?,?)",
            (str(r.patient_id), int(r.age), str(r.gender), str(r.condition), int(bool(r.reminder_enabled)))
        )
    for _, r in meds.iterrows():
        mid = str(r.medication_id)
        freq = str(r.frequency)
        times = _schedule_times(freq, "08:00,20:00" if "twice" in freq.lower() else "08:00")
        conn.execute(
            "INSERT OR IGNORE INTO medications(medication_id,patient_id,medication_name,dosage,frequency,time,instructions) VALUES(?,?,?,?,?,?,?)",
            (mid, str(r.patient_id), str(r.medication_name), str(r.dosage), freq, ",".join(times), "")
        )
    for _, r in adherence.iterrows():
        conn.execute(
            "INSERT OR REPLACE INTO adherence(patient_id,date,scheduled_doses,taken_doses,late_doses) VALUES(?,?,?,?,?)",
            (str(r.patient_id), str(r.date), int(r.scheduled_doses), int(r.taken_doses), int(r.late_doses))
        )


def upsert_patients(df: pd.DataFrame):
    initialize_database()
    conn = get_connection()
    for _, r in df.iterrows():
        conn.execute(
            "INSERT INTO patients(patient_id,age,gender,condition,reminder_enabled) VALUES(?,?,?,?,?)",
            (str(r.patient_id), int(r.age), str(r.gender), str(r.condition), int(r.reminder_enabled))
        )
    conn.commit(); conn.close()


def add_patient_user(conn, name, email, password_hash, age, gender, condition, reminder_enabled, timezone_name="Asia/Kolkata"):
    cur = conn.execute(
        "INSERT INTO users(patient_id,name,email,password_hash,initials,age,gender,condition,timezone,reminder_enabled) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (None, name, email, password_hash, "".join(x[0].upper() for x in name.split()[:2]) or "P", age, gender, condition, timezone_name, reminder_enabled)
    )
    user_id = cur.lastrowid
    patient_id = f"PT-{user_id:05d}"
    conn.execute("UPDATE users SET patient_id=? WHERE id=?", (patient_id, user_id))
    conn.execute(
        "INSERT INTO patients(patient_id,age,gender,condition,timezone,reminder_enabled,user_id) VALUES(?,?,?,?,?,?,?)",
        (patient_id, age, gender, condition, timezone_name, reminder_enabled, user_id)
    )
    return user_id, patient_id


def add_medication(conn, patient_id, name, dosage, frequency, time_text, instructions=""):
    times = _schedule_times(frequency, time_text)
    medication_id = f"{patient_id}-M{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    cur = conn.execute(
        "INSERT INTO medications(medication_id,patient_id,medication_name,dosage,frequency,time,instructions) VALUES(?,?,?,?,?,?,?)",
        (medication_id, patient_id, name, dosage, frequency, ",".join(times), instructions)
    )
    return cur.lastrowid


def log_audit(conn, actor_type, actor_id, action, entity_type=None, entity_id=None, details=None):
    conn.execute(
        "INSERT INTO audit_log(actor_type,actor_id,action,entity_type,entity_id,details) VALUES(?,?,?,?,?,?)",
        (actor_type, str(actor_id) if actor_id is not None else None, action, entity_type, str(entity_id) if entity_id is not None else None, json.dumps(details or {}, ensure_ascii=False))
    )


def patient_id_for_user(conn, user_id):
    row = conn.execute("SELECT patient_id FROM users WHERE id=?", (user_id,)).fetchone()
    return row["patient_id"] if row else None


def expected_slots_for_date(conn, patient_id, day):
    rows = conn.execute(
        "SELECT * FROM medications WHERE patient_id=? AND active=1 AND date(created_at) <= ? ORDER BY time",
        (patient_id, day.isoformat())
    ).fetchall()
    slots = []
    for med in rows:
        for t in _schedule_times(med["frequency"], med["time"]):
            slots.append((med, t))
    return slots


def refresh_patient_adherence(conn, patient_id, start_date=None, end_date=None):
    """Aggregate completed patient dose slots without treating snooze as missed."""
    if not patient_id:
        return
    today = date.today()
    if end_date is None:
        # Include today so a newly recorded dose is immediately visible in
        # adherence and analytics. Snoozed/upcoming slots are not counted as taken.
        end_date = today
    if start_date is None:
        first = conn.execute("SELECT MIN(date(created_at)) FROM medications WHERE patient_id=?", (patient_id,)).fetchone()[0]
        start_date = datetime.strptime(first, "%Y-%m-%d").date() if first else end_date
    if start_date > end_date:
        return
    day = start_date
    while day <= end_date:
        slots = expected_slots_for_date(conn, patient_id, day)
        if slots:
            taken = 0; late = 0
            for med, slot in slots:
                d = conn.execute(
                    "SELECT status,taken_at FROM doses WHERE user_id=(SELECT user_id FROM patients WHERE patient_id=?) AND medicine_id=? AND dose_date=? AND scheduled_time=?",
                    (patient_id, med["id"], day.isoformat(), slot)
                ).fetchone()
                if d and d["status"] == "taken":
                    taken += 1
                    if d["taken_at"]:
                        try:
                            taken_dt = datetime.fromisoformat(d["taken_at"])
                            sched_dt = datetime.fromisoformat(f"{day.isoformat()}T{slot}:00")
                            # Dose timestamps may be timezone-aware while the scheduled
                            # slot is stored without timezone information. Normalize both
                            # to the same local wall-clock representation before comparing.
                            if taken_dt.tzinfo is not None:
                                taken_dt = taken_dt.replace(tzinfo=None)
                            if sched_dt.tzinfo is not None:
                                sched_dt = sched_dt.replace(tzinfo=None)
                            if taken_dt > sched_dt + timedelta(minutes=30):
                                late += 1
                        except ValueError:
                            pass
            conn.execute(
                "INSERT OR REPLACE INTO adherence(patient_id,date,scheduled_doses,taken_doses,late_doses) VALUES(?,?,?,?,?)",
                (patient_id, day.isoformat(), len(slots), taken, late)
            )
        day += timedelta(days=1)


def refresh_all_patient_adherence(conn):
    patients = conn.execute("SELECT patient_id FROM patients WHERE user_id IS NOT NULL").fetchall()
    for row in patients:
        refresh_patient_adherence(conn, row["patient_id"])


def export_patients_csv(path):
    initialize_database()
    conn = get_connection()
    df = pd.read_sql_query("SELECT patient_id,age,gender,condition,reminder_enabled FROM patients ORDER BY patient_id", conn)
    conn.close(); df.to_csv(path, index=False)
    return df
