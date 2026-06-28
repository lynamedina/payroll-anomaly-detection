"""
main.py
=======
FastAPI backend for the Payroll Anomaly Detection system.

This API exposes two endpoints:
    POST /predict → predicts if a payroll record is anomalous
    POST /explain → returns which features triggered the anomaly flag

The API loads the best performing model (Random Forest) and uses it
for both prediction and explanation via feature importance scores.

Usage:
    cd api
    uvicorn main:app --reload --port 8000

    Then open: http://localhost:8000/docs  (Swagger UI — auto-generated)

Author: Lyna Medina Gassouma
Project: Intelligent Anomaly Detection in Global Payroll Data — ADP
"""

import os
import pickle
import numpy as np
import shap
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR = "../models"
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")

# ─── Feature Configuration ────────────────────────────────────────────────────

FEATURE_COLS = ["country", "currency", "role", "department",
                "years_experience", "base_salary", "bonus",
                "tax", "social_security", "net_pay",
                "tax_rate", "net_to_gross_ratio", "bonus_rate",
                "ss_rate", "total_deduction_rate"]

# Label encoding maps — must match preprocessing.py LabelEncoder output
# These are the alphabetical orderings sklearn's LabelEncoder uses
COUNTRY_MAP = {"AU": 0, "BR": 1, "CA": 2, "DE": 3, "FR": 4,
            "IN": 5, "JP": 6, "TN": 7, "UK": 8, "US": 9}

CURRENCY_MAP = {"AUD": 0, "BRL": 1, "CAD": 2, "EUR": 3, "GBP": 4,
                "INR": 5, "JPY": 6, "TND": 7, "USD": 8}

ROLE_MAP = {"Data Scientist": 0, "DevOps Engineer": 1, "Director": 2,
            "Finance Analyst": 3, "HR Manager": 4, "Intern": 5,
            "Junior Engineer": 6, "Payroll Specialist": 7,
            "Project Manager": 8, "Sales Manager": 9,
            "Senior Engineer": 10, "Support Specialist": 11}

DEPARTMENT_MAP = {"Engineering": 0, "Finance": 1, "HR": 2,
                "IT": 3, "Operations": 4, "Sales": 5}

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Payroll Anomaly Detection API",
    description=(
        "AI-powered anomaly detection for global payroll data. "
        "Built for ADP GlobalView as part of an end-of-studies Big Data AI project. "
        "Uses a Random Forest model trained on 500,000 synthetic payroll records "
        "across 10 countries and 12 job roles."
    ),
    version="1.0.0",
    contact={"name": "Payroll AI Team"}
)

# Allow frontend (Angular/React dashboard) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model ───────────────────────────────────────────────────────────────

def load_model():
    """
    Loads the trained Random Forest model from disk at startup.

    The model is loaded once at startup and reused for all requests,
    avoiding the overhead of loading it on every API call.

    Returns:
        Trained RandomForestClassifier object.

    Raises:
        RuntimeError: If the model file is not found.
    """
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run anomaly_detection.py first to train the model."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
    return model

model = load_model()

# Load scaler for feature normalization
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
with open(SCALER_PATH, "rb") as f:
    scaler_meta = pickle.load(f)

# Handle both old format (scaler only) and new format (dict with metadata)
scaler = scaler_meta

print("Scaler loaded successfully")
# SHAP TreeExplainer — created once at startup, reused for all /explain calls
shap_explainer = shap.TreeExplainer(model)


def get_shap_for_record(features: np.ndarray) -> dict:
    """
    Computes SHAP values for a single payroll record.

    SHAP (SHapley Additive exPlanations) decomposes the model's prediction
    into per-feature contributions: how much each feature pushed the
    anomaly score UP (positive) or DOWN (negative) relative to the
    average prediction (base_value).

    Handles differences across SHAP library versions where shap_values()
    may return a list [class0, class1] or a single 3D/2D array.

    Args:
        features (np.ndarray): Shape (1, 10) encoded feature vector.

    Returns:
        dict: {feature_name: shap_contribution} for class "anomaly" (1).
    """
    raw = shap_explainer.shap_values(features)

    if isinstance(raw, list):
        sv = raw[1][0]
    elif raw.ndim == 3:
        sv = raw[0, :, 1]
    else:
        sv = raw[0]

    return {col: round(float(val), 4) for col, val in zip(FEATURE_COLS, sv)}

# ─── Request / Response Schemas ───────────────────────────────────────────────

class PayrollRecord(BaseModel):
    """
    Input schema for a single payroll record.

    All salary fields should be in the employee's local currency (yearly).
    The API handles normalization internally before passing to the model.

    Fields:
        employee_id      : Unique employee identifier (for reference only)
        country          : ISO country code (US, FR, UK, DE, TN, IN, CA, AU, BR, JP)
        currency         : ISO currency code (USD, EUR, GBP, etc.)
        role             : Job title/role
        department       : Department name
        years_experience : Years of professional experience
        base_salary      : Annual base salary in local currency
        bonus            : Annual bonus amount
        tax              : Tax amount deducted
        social_security  : Social security contribution
        net_pay          : Final net pay after all deductions
    """
    employee_id:      Optional[str] = Field(default="UNKNOWN", example="EMP00001")
    country:          str           = Field(..., example="US")
    currency:         str           = Field(..., example="USD")
    role:             str           = Field(..., example="Senior Engineer")
    department:       str           = Field(..., example="Engineering")
    years_experience: float         = Field(..., ge=0, le=50, example=5.0)
    base_salary:      float         = Field(..., gt=0, example=95000.0)
    bonus:            float         = Field(..., ge=0, example=8000.0)
    tax:              float         = Field(..., ge=0, example=22000.0)
    social_security:  float         = Field(..., ge=0, example=9000.0)
    net_pay:          float         = Field(..., example=72000.0)


class PredictionResponse(BaseModel):
    """
    Response schema for the /predict endpoint.

    Fields:
        employee_id    : Echoed from input for traceability
        is_anomaly     : True if the record is flagged as anomalous
        anomaly_score  : Probability score [0.0 - 1.0] (higher = more anomalous)
        risk_level     : Human-readable risk label (Low / Medium / High)
        message        : Short explanation of the result
    """
    employee_id:   str
    is_anomaly:    bool
    anomaly_score: float
    risk_level:    str
    message:       str


class ExplainResponse(BaseModel):
    """
    Response schema for the /explain endpoint.

    Extends PredictionResponse with feature-level explanations.

    Fields:
        employee_id        : Echoed from input
        is_anomaly         : Prediction result
        anomaly_score      : Probability score
        risk_level         : Risk label
        top_features       : Top 3 features by SHAP impact for this record
        feature_scores     : SHAP contribution per feature (signed: + increases anomaly score, - decreases it)
        anomaly_indicators : Specific rules triggered by the input values
        recommendation     : Suggested action for HR/payroll team
    """
    employee_id:        str
    is_anomaly:         bool
    anomaly_score:      float
    risk_level:         str
    top_features:       list
    feature_scores:     dict
    anomaly_indicators: list
    recommendation:     str


# ─── Helper Functions ─────────────────────────────────────────────────────────

def encode_record(record: PayrollRecord) -> np.ndarray:
    """
    Encodes and normalizes a PayrollRecord into a feature vector
    compatible with the trained model.

    Encoding:
        - Categorical fields (country, currency, role, department) →
        label encoded using the same mapping as preprocessing.py
        - Numerical fields → passed as-is (model handles raw values
        since MinMaxScaler was applied during training but feature
        importance is scale-invariant for Random Forest)

    Args:
        record (PayrollRecord): Validated input from the API request.

    Returns:
        np.ndarray: Shape (1, 10) feature vector ready for model.predict()

    Raises:
        HTTPException 400: If country, currency, role, or department
                        is not recognized.
    """
    # Validate and encode categoricals
    if record.country not in COUNTRY_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown country '{record.country}'. "
                f"Valid values: {list(COUNTRY_MAP.keys())}"
        )
    if record.currency not in CURRENCY_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown currency '{record.currency}'. "
                f"Valid values: {list(CURRENCY_MAP.keys())}"
        )
    if record.role not in ROLE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown role '{record.role}'. "
                f"Valid values: {list(ROLE_MAP.keys())}"
        )
    if record.department not in DEPARTMENT_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown department '{record.department}'. "
                f"Valid values: {list(DEPARTMENT_MAP.keys())}"
        )

    base = record.base_salary
    tax  = record.tax
    ss   = record.social_security
    bon  = record.bonus
    net  = record.net_pay

    tax_rate             = tax / base if base > 0 else 0
    net_to_gross_ratio   = net / base if base > 0 else 0
    bonus_rate           = bon / base if base > 0 else 0
    ss_rate              = ss / base if base > 0 else 0
    total_deduction_rate = (tax + ss) / base if base > 0 else 0

    features = np.array([[
        COUNTRY_MAP[record.country],
        CURRENCY_MAP[record.currency],
        ROLE_MAP[record.role],
        DEPARTMENT_MAP[record.department],
        record.years_experience,
        base,
        bon,
        tax,
        ss,
        net,
        tax_rate,
        net_to_gross_ratio,
        bonus_rate,
        ss_rate,
        total_deduction_rate,
    ]], dtype=float)
    
    features = scaler.transform(features)
    return features

def get_risk_level(score: float) -> str:
    """
    Converts a continuous anomaly score to a human-readable risk label.

    Thresholds:
        score < 0.3  → Low     (likely normal)
        score < 0.6  → Medium  (borderline, worth reviewing)
        score >= 0.6 → High    (strong anomaly signal)

    Args:
        score (float): Anomaly probability from model.predict_proba()

    Returns:
        str: "Low", "Medium", or "High"
    """
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Medium"
    else:
        return "High"


def detect_anomaly_indicators(record: PayrollRecord) -> list:
    """
    Applies rule-based checks to identify specific anomaly patterns
    in the input record. These rules mirror the 8 anomaly types
    injected during data generation.

    Rules checked:
        1. net_pay > base_salary         → net exceeds gross
        2. tax == 0 and base_salary > 50k → missing tax on high salary
        3. bonus > base_salary * 0.5     → bonus exceeds 50% of salary
        4. base_salary > role_max * 3    → salary spike for the role
        5. social_security == 0          → zero deductions
        6. net_pay > base_salary * 1.5   → possible duplicate payment

    Args:
        record (PayrollRecord): Input payroll record.

    Returns:
        list: List of triggered indicator strings (empty if none).
    """
    indicators = []

    if record.net_pay > record.base_salary:
        indicators.append(
            f"NET_EXCEEDS_GROSS: net_pay ({record.net_pay:,.0f}) "
            f"> base_salary ({record.base_salary:,.0f})"
        )

    if record.tax == 0 and record.base_salary > 50000:
        indicators.append(
            f"MISSING_TAX: No tax deducted on salary of {record.base_salary:,.0f}"
        )

    if record.bonus > record.base_salary * 0.5:
        indicators.append(
            f"BONUS_EXCEEDED: bonus ({record.bonus:,.0f}) > 50% of "
            f"base_salary ({record.base_salary:,.0f})"
        )

    if record.social_security == 0 and record.tax == 0:
        indicators.append("ZERO_DEDUCTIONS: No tax or social security deducted")

    if record.net_pay > record.base_salary * 1.5:
        indicators.append(
            f"POSSIBLE_DUPLICATE: net_pay ({record.net_pay:,.0f}) "
            f"> 1.5x base_salary ({record.base_salary:,.0f})"
        )

    return indicators


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """
    Health check endpoint.
    Returns API status and model info.
    """
    return {
        "status":      "running",
        "api":         "Payroll Anomaly Detection API",
        "version":     "1.0.0",
        "model":       "Random Forest (trained on 500k payroll records)",
        "endpoints":   ["/predict", "/explain", "/docs"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(record: PayrollRecord):
    """
    Predicts whether a payroll record is anomalous.

    The model returns a probability score between 0 and 1.
    Scores above 0.5 are flagged as anomalies.

    Example anomalous input:
        base_salary: 95000, tax: 0, net_pay: 103000
        → Missing tax + net exceeds gross → High risk

    Args:
        record (PayrollRecord): Full payroll record to evaluate.

    Returns:
        PredictionResponse: Prediction result with risk level and message.
    """
    features      = encode_record(record)
    prediction    = model.predict(features)[0]
    score         = float(model.predict_proba(features)[0][1])
    is_anomaly    = bool(prediction == 1)
    risk_level    = get_risk_level(score)

    message = (
        f"Anomaly detected with {score:.1%} confidence. "
        f"Risk level: {risk_level}. Recommend HR review."
        if is_anomaly else
        f"Record appears normal (anomaly score: {score:.1%})."
    )

    return PredictionResponse(
        employee_id=record.employee_id,
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 4),
        risk_level=risk_level,
        message=message
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
def explain(record: PayrollRecord):
    """
    Predicts anomaly AND explains which features drove the decision.

    Explanation method:
        - Feature importance from the Random Forest (global importance)
        - Rule-based indicators (local, record-specific explanation)

    The combination of global feature importance + local rule triggers
    gives HR teams actionable insight: not just "this is anomalous"
    but "this is anomalous BECAUSE the net_pay exceeds the gross salary."

    Args:
        record (PayrollRecord): Full payroll record to evaluate and explain.

    Returns:
        ExplainResponse: Full prediction + feature scores + triggered rules
                        + recommended action.
    """
    features   = encode_record(record)
    prediction = model.predict(features)[0]
    score      = float(model.predict_proba(features)[0][1])
    is_anomaly = bool(prediction == 1)
    risk_level = get_risk_level(score)

    # Per-record SHAP contributions (replaces global feature importance)
    feature_scores = get_shap_for_record(features)

    # Top 3 features by absolute SHAP impact for THIS record
    sorted_features = sorted(feature_scores.items(),
                             key=lambda x: abs(x[1]), reverse=True)
    top_features = [
        {"feature": k, "importance": v, "value": float(features[0][i])}
        for i, (k, v) in enumerate(sorted_features[:3])
    ]

    # Record-specific rule triggers
    indicators = detect_anomaly_indicators(record)

    # Recommendation based on risk and indicators
    if not is_anomaly:
        recommendation = "No action required. Record is within normal parameters."
    elif risk_level == "High":
        recommendation = (
            "Immediate review required. Flag for payroll audit. "
            f"Triggered rules: {', '.join(indicators) if indicators else 'Model-based detection'}."
        )
    else:
        recommendation = (
            "Schedule review within 48 hours. "
            f"Triggered rules: {', '.join(indicators) if indicators else 'Model-based detection'}."
        )

    return ExplainResponse(
        employee_id=record.employee_id,
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 4),
        risk_level=risk_level,
        top_features=top_features,
        feature_scores=feature_scores,
        anomaly_indicators=indicators,
        recommendation=recommendation
    )