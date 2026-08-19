Models folder

train_model.py:
    Reproducibly generates the bundled synthetic/demo adherence history and trains
    the leakage-safe GradientBoostingClassifier using a patient-level 80/20 holdout.

adherence_model.pkl:
    Trained GradientBoostingClassifier used by the shared prediction service.

model.pkl:
    Same trained model retained for training/evaluation workflows.

feature_columns.pkl:
    Exact feature order expected by the prediction module.

prediction_config.pkl:
    Prediction threshold and evaluation configuration.

model_metrics.csv / model_metrics.pkl:
    Verified held-out evaluation metrics.

Current verified holdout accuracy: 89.0%
Current verified holdout ROC-AUC: 92.07%

To retrain after changing the dataset:
    python models/train_model.py

Required packages:
    pandas
    scikit-learn
    joblib

Important: the bundled dataset is synthetic/demo data and these metrics are not
clinical validation.
