UTILS FOLDER

preprocessing.py
----------------
Loads, cleans, and prepares the CSV data for the ML model.

Main functions:
- load_data()
- clean_adherence_data()
- build_patient_features()
- get_patient_history()
- get_adherence_summary()

database.py
-----------
Provides SQLite database support.

Main functions:
- get_connection()
- initialize_database()
- save_dataframe()
- load_table()
- save_patient_summary()
- load_patient_summary()
- import_csv_data()

The utilities expect the project structure:

medication_adherence/
├── data/
│   ├── patients.csv
│   ├── medications.csv
│   └── adherence.csv
└── utils/
    ├── preprocessing.py
    └── database.py
