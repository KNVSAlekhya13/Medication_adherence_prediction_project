# MediTrack Research Dashboard

The Streamlit dashboard analyzes the **synthetic research/demo dataset** in `admin_app/data/`. It is intentionally isolated from the operational Flask patient database.

## Research modules

- Adherence measurement
- Non-adherence risk estimation
- Threshold-based adherence factor screening
- Pattern/anomaly detection
- Backtested exploratory trend forecasting
- Rule-based adherence-support suggestions
- Non-causal what-if scenarios
- Research reports

## Important terminology

Do not call the factor module "statistical factor analysis"; it is transparent rule-based screening.

Do not call the intervention module an ML recommender; it is a rule-based support layer.

Do not call the forecast a clinical forecast; the dashboard compares recent-mean and linear-trend baselines using rolling backtesting.

Do not call the risk a clinical risk score. It is the model-estimated probability of future adherence below 70%.

## Run

```powershell
pip install -r requirements.txt
python models/train_model.py
streamlit run dashboard/dashboard.py
```

## Dataset

The included dataset is synthetic/demo data. It is not real patient data and is not evidence of clinical effectiveness.

## Model artifacts

Only the following model artifacts are required:

- `adherence_model.pkl`
- `feature_columns.pkl`
- `prediction_config.pkl`
- `model_metrics.csv`

Unused legacy scaler/duplicate metric artifacts were removed.

## Limitations

Research outputs require external validation on appropriately governed, representative data before any scientific or clinical claim.
