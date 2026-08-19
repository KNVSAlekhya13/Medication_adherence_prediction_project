"""Compatibility wrapper around the single shared MediTrack SQLite database."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from shared.db import DB_PATH, get_connection, initialize_database, export_patients_csv, upsert_patients, refresh_all_patient_adherence

DEFAULT_DB=DB_PATH

def initialize_database_legacy(db_path=None):
    return initialize_database(import_csv=True)

def get_connection_legacy(db_path=None):
    return get_connection()

# Backward-compatible names used by older code.
initialize_database = initialize_database
get_connection = get_connection
