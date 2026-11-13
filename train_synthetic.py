"""Retrain on synthetic data where social flags ARE predictive."""
import numpy as np, pandas as pd, xgboost as xgb, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
N = 6000

age             = rng.integers(18, 75, N)
sex_male        = rng.integers(0, 2, N)
vehicle_risk    = rng.uniform(0.7, 1.4, N)
current_premium = rng.integers(400, 4000, N)
has_claims      = rng.integers(0, 2, N)
annual_mileage  = rng.integers(5, 40, N)
social_profiles = rng.integers(0, 6, N)
driving_flags   = np.where(rng.random(N) < 0.15, rng.integers(1, 3, N), 0)
alcohol_flags   = np.where(rng.random(N) < 0.08, rng.integers(1, 2, N), 0)
speeding_flags  = np.where(rng.random(N) < 0.12, rng.integers(1, 3, N), 0)
social_risk     = np.minimum(1.0, 1 - np.exp(-0.7 * (driving_flags + alcohol_flags + speeding_flags)))
social_conf     = np.where(social_profiles > 0, rng.uniform(0.5, 0.95, N), 0.0)

logit = (
    -2.5 + 0.04 * (age - 40) + 0.30 * sex_male
    + 1.20 * (vehicle_risk - 1.0) - 0.0005 * (current_premium - 1200)
    + 0.80 * has_claims + 0.02 * (annual_mileage - 15)
    + 1.80 * driving_flags + 2.20 * alcohol_flags + 1.50 * speeding_flags
    + 0.60 * social_risk
)
prob = 1 / (1 + np.exp(-logit))
y = (rng.random(N) < prob).astype(int)

X = pd.DataFrame({
    "age": age, "sex_male": sex_male, "vehicle_risk": vehicle_risk,
    "current_premium": current_premium, "has_claims_recent": has_claims,
    "annual_mileage_k": annual_mileage,
    "social_risk_score": social_risk,
    "social_driving_flags": driving_flags,
    "social_alcohol_flags": alcohol_flags,
    "social_speeding_flags": speeding_flags,
    "social_profiles_provided": social_profiles,
    "social_confidence_avg": social_conf,
})

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="auc")
m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
print("AUC:", roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)

joblib.dump(m, "artifacts/xgb_risk_v1.joblib")
joblib.dump(lr, "artifacts/lr_calibrator_v1.joblib")
joblib.dump(sc, "artifacts/scaler_v1.joblib")
print("✅ Artifacts saved.")



