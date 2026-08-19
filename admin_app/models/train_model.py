"""MediTrack synthetic research model training (reproducible, leakage-safe)."""
from pathlib import Path
import csv, json
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'admin_app/data'; MODELS=ROOT/'admin_app/models'; MODELS.mkdir(exist_ok=True)
SEED=42
rng=np.random.RandomState(SEED)
patients=pd.read_csv(DATA/'patients.csv')

# Synthetic/demo adherence history. The data is intentionally synthetic and is not
# clinical evidence. Each patient has a latent adherence regime, but test patients
# remain completely unseen during fitting and threshold selection.
rows=[]
for _, p in patients.iterrows():
    low = rng.rand() < 0.38
    base = rng.beta(30*0.55, 30*(1-0.55)) if low else rng.beta(30*0.93, 30*(1-0.93))
    for d in range(35):
        taken = int(rng.rand() < base)
        late = int(taken and rng.rand() < 0.10)
        day = pd.Timestamp('2026-01-01') + pd.Timedelta(days=d)
        rows.append([p['patient_id'], day.strftime('%Y-%m-%d'), 1, taken, late])
adh = pd.DataFrame(rows, columns=['patient_id','date','scheduled_doses','taken_doses','late_doses'])
adh.to_csv(DATA/'adherence.csv', index=False)

# Features use only the first 24 days. The final 11 days define the future target.
rows=[]
for pid, g in adh.groupby('patient_id'):
    g=g.sort_values('date').reset_index(drop=True)
    h=g.iloc[:24]; future=g.iloc[24:]
    hist=float(h.taken_doses.mean()); recent=float(h.tail(7).taken_doses.mean())
    missed=int(24-h.taken_doses.sum()); late=int(h.late_doses.sum())
    miss_rate=float(missed/24); recent_missed=int(7-h.tail(7).taken_doses.sum())
    target=int(future.taken_doses.mean() >= 0.70)
    rows.append([pid,hist,recent,missed,late,miss_rate,recent_missed,target])
F=pd.DataFrame(rows, columns=['patient_id','hist','recent','missed','late','miss_rate','recent_missed','target'])
P=patients[['patient_id','age','reminder_enabled']]
D=F.merge(P,on='patient_id')
features=['age','reminder_enabled','hist','recent','missed','late','miss_rate','recent_missed']
X=D[features]; y=D['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=SEED,stratify=y)

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
base=GradientBoostingClassifier(n_estimators=200,max_depth=2,learning_rate=.05,random_state=SEED)
p_cv=cross_val_predict(base,X_train,y_train,cv=cv,method='predict_proba')[:,1]
threshold,cv_acc=max(((float(t),accuracy_score(y_train,p_cv>=t)) for t in np.linspace(.10,.90,801)),key=lambda x:x[1])
model=GradientBoostingClassifier(n_estimators=200,max_depth=2,learning_rate=.05,random_state=SEED)
model.fit(X_train,y_train)
p=model.predict_proba(X_test)[:,1]; pred=(p>=threshold).astype(int)

metrics={
 'accuracy':float(accuracy_score(y_test,pred)),
 'precision':float(precision_score(y_test,pred,zero_division=0)),
 'recall':float(recall_score(y_test,pred,zero_division=0)),
 'f1_score':float(f1_score(y_test,pred,zero_division=0)),
 'roc_auc':float(roc_auc_score(y_test,p)),
 'cv_accuracy':float(cv_acc),
 'n_patients':int(len(D)), 'n_train':int(len(X_train)), 'n_test':int(len(X_test)),
 'positive_rate':float(y.mean()), 'threshold':float(threshold),
 'threshold_selection':'Maximum accuracy on stratified CV predictions from training split',
 'training_sklearn_version':__import__('sklearn').__version__,
 'model_family':'GradientBoostingClassifier',
 'evaluation':'Patient-level 80/20 stratified holdout; threshold selected only from training CV; synthetic/demo data.'
}
for k in ['accuracy','precision','recall','f1_score','roc_auc','cv_accuracy']:
    metrics[k+'_percent']=metrics[k]*100

joblib.dump(model,MODELS/'adherence_model.pkl')
joblib.dump(model,MODELS/'model.pkl')
joblib.dump(features,MODELS/'feature_columns.pkl')
joblib.dump({'prediction_threshold':float(threshold),'minimum_history_days':7,
             'target_definition':'future adherence >= 70%; protocol threshold is TARGET_ADHERENCE=0.70',
             'feature_cutoff':'Only records on/before prediction cutoff are used in production.',
             'evaluation':metrics['evaluation'],'training_data':'Bundled synthetic/demo CSV only; live patient data is not used for retraining.',
             'sklearn_version':__import__('sklearn').__version__,'model_family':'GradientBoostingClassifier',
             'warning':'Synthetic/demo data; metrics are not clinical validation.'}, MODELS/'prediction_config.pkl')
joblib.dump(metrics,MODELS/'model_metrics.pkl')
with open(MODELS/'model_metrics.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics.keys())); w.writeheader(); w.writerow(metrics)
with open(MODELS/'candidate_cv_results.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['model','cv_accuracy']); w.writerow(['GradientBoostingClassifier',cv_acc])
print(json.dumps(metrics,indent=2))
