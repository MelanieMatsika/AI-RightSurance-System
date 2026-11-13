"""
AI RightSurance – Risk Scoring Engine (FastAPI)
Run:  uvicorn risk_engine:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Literal
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import datetime
import re
import os
from datetime import timezone

# ═══════════════════════════════════════════════════════════════
# 1. CONTRACTS
# ═══════════════════════════════════════════════════════════════
class SocialHandles(BaseModel):
    facebook: Optional[str] = ""
    instagram: Optional[str] = ""
    tiktok: Optional[str] = ""
    linkedin: Optional[str] = ""
    x: Optional[str] = ""


class ApplicantPayload(BaseModel):
    idNumber: str
    name: str
    surname: str
    sex: str
    cell: str
    email: EmailStr
    address: str
    insuranceCompany: str
    vehicleMake: str
    vehicleModel: str
    registration: str
    carColor: str
    currentPremium: str
    social: SocialHandles


class RiskResult(BaseModel):
    overallScore: int
    rating: Literal["Excellent", "Good", "Fair", "Moderate", "High Risk"]
    lifestyleScore: int
    drivingScore: int
    socialActivityScore: int
    recommendedPremium: int
    premiumAdjustmentPct: float
    confidence: float
    reasons: List[str]
    socialEvidence: List[dict]
    modelVersion: str
    disclaimer: str


# ═══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
VEHICLE_RISK = {
    "toyota": 0.85, "vw": 0.95, "bmw": 1.25, "mercedes": 1.20,
    "ford": 0.90, "hyundai": 0.88, "kia": 0.87, "audi": 1.15,
}

FEATURE_ORDER = [
    "age", "sex_male", "vehicle_risk", "current_premium",
    "has_claims_recent", "annual_mileage_k",
    "social_risk_score", "social_driving_flags", "social_alcohol_flags",
    "social_speeding_flags", "social_profiles_provided", "social_confidence_avg",
]


def parse_premium(p: str) -> float:
    return float(re.sub(r"[^\d.]", "", p or "0") or 0)


def age_from_sa_id(id_number: str) -> int:
    if len(id_number) != 13 or not id_number.isdigit():
        return 35
    yy = int(id_number[0:2])
    now = datetime.date.today()
    year = 2000 + yy if yy < (now.year % 100) else 1900 + yy
    try:
        dob = datetime.date(year, int(id_number[2:4]), 1)
    except ValueError:
        return 35
    return max(18, (now - dob).days // 365)


def build_features(p: ApplicantPayload, social: dict) -> pd.DataFrame:
    make = p.vehicleMake.lower().strip()
    veh_risk = VEHICLE_RISK.get(make, 1.0)
    return pd.DataFrame([{
        "age":                      age_from_sa_id(p.idNumber),
        "sex_male":                 int(p.sex.lower() == "male"),
        "vehicle_risk":             veh_risk,
        "current_premium":          parse_premium(p.currentPremium),
        "has_claims_recent":        0,
        "annual_mileage_k":         15,
        "social_risk_score":        social["risk_score"],
        "social_driving_flags":     social["driving_flags"],
        "social_alcohol_flags":     social["alcohol_flags"],
        "social_speeding_flags":    social["speeding_flags"],
        "social_profiles_provided": social["profiles_provided"],
        "social_confidence_avg":    social["confidence_avg"],
    }])[FEATURE_ORDER]


# ═══════════════════════════════════════════════════════════════
# 3. MODEL LOADING
# ═══════════════════════════════════════════════════════════════
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_VERSION = "rightsurance-risk-1.0.0"


def _load_or_bootstrap():
    xgb_path = os.path.join(ARTIFACT_DIR, "xgb_risk_v1.joblib")
    lr_path = os.path.join(ARTIFACT_DIR, "lr_calibrator_v1.joblib")
    sc_path = os.path.join(ARTIFACT_DIR, "scaler_v1.joblib")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if all(os.path.exists(p) for p in (xgb_path, lr_path, sc_path)):
        print("✅ Loaded trained model artifacts.")
        return joblib.load(xgb_path), joblib.load(lr_path), joblib.load(sc_path)

    print("⚠️  No trained artifacts — using bootstrap model.")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, len(FEATURE_ORDER)))
    y = (X[:, 6] * 0.5 + rng.normal(scale=0.5, size=200) > 0.3).astype(int)
    m = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="auc")
    m.fit(X, y)
    lr = LogisticRegression().fit(X, y)
    sc = StandardScaler().fit(X)
    joblib.dump(m, xgb_path)
    joblib.dump(lr, lr_path)
    joblib.dump(sc, sc_path)
    return m, lr, sc


xgb_model, lr_model, scaler = _load_or_bootstrap()


# ═══════════════════════════════════════════════════════════════
# 4. SOCIAL MEDIA ANALYSIS
# ═══════════════════════════════════════════════════════════════
SOCIAL_CONFIDENCE_FLOOR = 0.65

RISK_KEYWORDS = {
    "driving":  ["drift", "street race", "burnout", "reckless", "racing",
                 "donut", "cut off", "road rage"],
    "alcohol":  ["drunk", "drinking and driving", "booze", "tipsy driving",
                 "after a few beers", "DUI", "DWI"],
    "speeding": ["speeding", "km/h", "mph", "racing", "high speed", "fast lane"],
}

_nlp = None


def _load_nlp():
    if os.getenv("MODEL_OFFLINE") == "1":
        return None
    try:
        from transformers import pipeline
        print("✅ Loading NLP zero-shot classifier...")
        return pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli", device=-1)
    except Exception as e:
        print(f"⚠️  NLP unavailable, using keyword matching: {e}")
        return None


def _ensure_nlp():
    global _nlp
    if _nlp is None:
        _nlp = _load_nlp()
    return _nlp


def _fetch_public_posts(platform: str, handle_or_url: str) -> list:
    """
    Fetch public posts for a handle.
    PRODUCTION: replace with Meta Graph API / X API / TikTok API calls using OAuth tokens.
    DEV: returns a mock so the pipeline runs end-to-end.
    """
    print(f"  → Fetching public posts for {platform}: {handle_or_url}")
    return [{
        "text": f"Sample post from {platform} handle {handle_or_url}",
        "image_url": None,
        "url": f"https://{platform}.com/{handle_or_url.lstrip('@')}/post/1",
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
    }]


def _classify_text(text: str) -> tuple:
    if not text or len(text.strip()) < 4:
        return None, 0.0
    tl = text.lower()
    hits = {cat: sum(1 for w in words if w in tl) for cat, words in RISK_KEYWORDS.items()}
    best_cat = max(hits, key=hits.get)
    best_count = hits[best_cat]

    nlp = _ensure_nlp()
    if nlp is not None:
        try:
            r = nlp(text, candidate_labels=[
                "dangerous driving", "drinking and driving",
                "speeding", "normal lifestyle"])
            label, score = r["labels"][0], float(r["scores"][0])
            if score >= SOCIAL_CONFIDENCE_FLOOR:
                if label == "dangerous driving":     return "driving", score
                if label == "drinking and driving":  return "alcohol", score
                if label == "speeding":              return "speeding", score
        except Exception as e:
            print(f"  ⚠️  NLP failed: {e}")

    if best_count >= 2: return best_cat, 0.70
    if best_count == 1: return best_cat, 0.66
    return None, 0.0


def _classify_image(image_url: str) -> tuple:
    if not image_url:
        return None, 0.0
    try:
        from transformers import pipeline
        from PIL import Image
        import io, requests
        vision = pipeline("image-classification",
                          model="google/vit-base-patch16-224", device=-1)
        img = Image.open(io.BytesIO(requests.get(image_url, timeout=8).content)).convert("RGB")
        result = vision(img, top_k=5)
        labels = {r["label"].lower(): r["score"] for r in result}
        for cat, hints in {
            "driving":  ["sports car", "racing", "convertible", "motorcycle"],
            "alcohol":  ["beer", "wine", "cocktail", "bar"],
            "speeding": ["race car", "sports car", "highway"],
        }.items():
            for h in hints:
                if h in labels and labels[h] >= SOCIAL_CONFIDENCE_FLOOR:
                    return cat, labels[h]
    except Exception as e:
        print(f"  ⚠️  Vision failed: {e}")
    return None, 0.0


def analyse_social(handles: SocialHandles) -> dict:
    profiles = {k: v for k, v in handles.dict().items() if v and v.strip()}
    flags = {"driving": 0, "alcohol": 0, "speeding": 0}
    confidences, evidence = [], []

    for platform, handle in profiles.items():
        posts = _fetch_public_posts(platform, handle)
        print(f"  📡 {platform}: {len(posts)} posts fetched")
        for post in posts:
            cat, conf = _classify_text(post.get("text", ""))
            if cat and conf >= SOCIAL_CONFIDENCE_FLOOR:
                flags[cat] += 1
                confidences.append(conf)
                evidence.append({"platform": platform, "url": post["url"],
                                 "confidence": round(conf, 3),
                                 "risk_category": cat,
                                 "timestamp": post["timestamp"]})
            cat, conf = _classify_image(post.get("image_url"))
            if cat and conf >= SOCIAL_CONFIDENCE_FLOOR:
                flags[cat] += 1
                confidences.append(conf)
                evidence.append({"platform": platform, "url": post["url"],
                                 "confidence": round(conf, 3),
                                 "risk_category": f"{cat}_image",
                                 "timestamp": post["timestamp"]})

    total = flags["driving"] + flags["alcohol"] + flags["speeding"]
    risk_score = 0.0 if total == 0 else min(1.0, 1 - np.exp(-0.7 * total))

    return {
        "risk_score": float(risk_score),
        "driving_flags": flags["driving"],
        "alcohol_flags": flags["alcohol"],
        "speeding_flags": flags["speeding"],
        "profiles_provided": len(profiles),
        "confidence_avg": float(np.mean(confidences)) if confidences else 0.0,
        "evidence": evidence,
    }


# ═══════════════════════════════════════════════════════════════
# 5. SCORING HELPERS
# ═══════════════════════════════════════════════════════════════
def rating_from_score(s: int) -> str:
    if s >= 85: return "Excellent"
    if s >= 70: return "Good"
    if s >= 55: return "Fair"
    if s >= 40: return "Moderate"
    return "High Risk"


def premium_from_score(base: int, s: int):
    if s >= 85: return round(base * 0.70), -30.0
    if s >= 70: return round(base * 0.85), -15.0
    if s >= 55: return base, 0.0
    if s >= 40: return round(base * 1.25), +25.0
    return round(base * 1.50), +50.0


FEATURE_LABELS = {
    "age": "Driver age", "sex_male": "Applicant sex",
    "vehicle_risk": "Vehicle make/model risk",
    "current_premium": "Current premium",
    "has_claims_recent": "Recent claims history",
    "annual_mileage_k": "Estimated annual mileage",
    "social_risk_score": "Social media risk indicator",
    "social_driving_flags": "Dangerous-driving content flags",
    "social_alcohol_flags": "Alcohol-related content flags",
    "social_speeding_flags": "Speeding content flags",
    "social_profiles_provided": "Social profiles provided",
    "social_confidence_avg": "AI confidence in social findings",
}


def build_reasons(model, X: pd.DataFrame, top_k: int = 4) -> List[str]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return ["Risk score derived from AI model."]
    ranked = sorted(zip(X.columns, imp), key=lambda t: -t[1])[:top_k]
    return [f"{FEATURE_LABELS.get(c, c)} contributed to this assessment." for c, _ in ranked]


# ═══════════════════════════════════════════════════════════════
# 6. FASTAPI APP
# ═══════════════════════════════════════════════════════════════
app = FastAPI(title="AI RightSurance Risk Engine", version=MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://client.rightsurance.co.za"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_VERSION}


@app.post("/api/v1/score", response_model=RiskResult)
def score(payload: ApplicantPayload):
    try:
        social = analyse_social(payload.social)
        X = build_features(payload, social)

        Xs = scaler.transform(X.values)
        risk_prob = float(xgb_model.predict_proba(X.values)[0, 1])
        calibrated = float(lr_model.predict_proba(Xs)[0, 1])
        blended = 0.7 * risk_prob + 0.3 * calibrated

        overall = max(0, min(100, int(round((1 - blended) * 100))))

        lifestyle = int(round(100 * (
            0.5 * (1 - min(social["risk_score"], 1))
            + 0.3 * (1 if X["age"].iloc[0] >= 25 else 0.6)
            + 0.2 * (1 - social["confidence_avg"])
        )))
        driving = int(round(100 * (1 - min(
            social["driving_flags"] * 0.25 + social["speeding_flags"] * 0.2, 1))))
        social_activity = int(round(100 * (1 - min(social["risk_score"], 1))))

        base = 1200
        premium, adj = premium_from_score(base, overall)
        reasons = build_reasons(xgb_model, X)
        if social["evidence"]:
            reasons.insert(0,
                f"{len(social['evidence'])} social-media item(s) flagged with average "
                f"AI confidence {social['confidence_avg']:.0%}.")

        return RiskResult(
            overallScore=overall,
            rating=rating_from_score(overall),
            lifestyleScore=lifestyle,
            drivingScore=driving,
            socialActivityScore=social_activity,
            recommendedPremium=premium,
            premiumAdjustmentPct=adj,
            confidence=round(float(1 - abs(blended - 0.5) * 2), 3),
            reasons=reasons,
            socialEvidence=social["evidence"],
            modelVersion=MODEL_VERSION,
            disclaimer=(
                "This score is an AI-generated indicator, not a final decision. "
                "All social-media findings require human verification and are used "
                "solely within POPIA/GDPR-compliant underwriting."
            ),
        )
    except Exception as e:
        raise HTTPException(500, f"Scoring failed: {e}")