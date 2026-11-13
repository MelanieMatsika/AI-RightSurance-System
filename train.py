"""Offline trainer. Run when you have labelled underwriting data."""
import pandas as pd, xgboost as xgb, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("data/underwriting_claims_v3.parquet")
FEATURES = [
    "age", "sex_male", "vehicle_risk", "current_premium",
    "has_claims_recent", "annual_mileage_k",
    "social_risk_score", "social_driving_flags", "social_alcohol_flags",
    "social_speeding_flags", "social_profiles_provided", "social_confidence_avg",
]
X = df[FEATURES]; y = df["label"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", use_label_encoder=False,
)
model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
print("AUC:", roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))

scaler = StandardScaler().fit(Xtr)
lr = LogisticRegression().fit(scaler.transform(Xtr), ytr)

joblib.dump(model, "artifacts/xgb_risk_v1.joblib")
joblib.dump(lr,    "artifacts/lr_calibrator_v1.joblib")
joblib.dump(scaler, "artifacts/scaler_v1.joblib")