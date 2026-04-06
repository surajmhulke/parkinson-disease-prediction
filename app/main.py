"""
app/main.py
-----------
FastAPI prediction service for Parkinson's disease classification.

Run:
    cd ensemble_project
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /             → health check
    POST /predict      → predict from JSON feature values
    GET  /features     → list required feature names
"""

import os
import sys
import json
import pickle
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
FEATURE_PATH = os.path.join(OUTPUTS_DIR, "feature_names.json")
DEFAULT_MODEL = "logistic_regression"


# ---------------------------------------------------------------------------
# Load artefacts at startup
# ---------------------------------------------------------------------------

def _load_feature_names() -> list[str]:
    if not os.path.exists(FEATURE_PATH):
        raise RuntimeError(
            "feature_names.json not found. Run 'python src/train.py' first."
        )
    with open(FEATURE_PATH) as f:
        return json.load(f)


def _load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise RuntimeError(
            f"Model '{name}' not found at {path}. Run 'python src/train.py' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# Lazy-load on first request to avoid startup crash when models aren't built yet
_feature_names: list[str] | None = None
_models: dict = {}


def get_feature_names() -> list[str]:
    global _feature_names
    if _feature_names is None:
        _feature_names = _load_feature_names()
    return _feature_names


def get_model(name: str):
    if name not in _models:
        _models[name] = _load_model(name)
    return _models[name]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Parkinson's Disease Prediction API",
    description=(
        "Predict whether a patient has Parkinson's disease based on "
        "voice measurement features. Trained on the Parkinson's Disease "
        "dataset using Logistic Regression, Random Forest, and XGBoost."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_EXAMPLE_FEATURES = {
    "tqwt_kurtosisValue_dec_34": 2.6202,
    "tqwt_kurtosisValue_dec_28": 1.562,
    "tqwt_kurtosisValue_dec_26": 1.6058,
    "tqwt_kurtosisValue_dec_25": 2.0062,
    "tqwt_maxValue_dec_1": 0.01432,
    "tqwt_minValue_dec_12": -0.024286,
    "tqwt_TKEO_mean_dec_32": 0.000013,
    "tqwt_TKEO_mean_dec_13": 0.000128,
    "tqwt_entropy_log_dec_26": -4459.3054,
    "tqwt_entropy_shannon_dec_35": 0.015562,
    "tqwt_entropy_shannon_dec_17": 360.7126,
    "tqwt_entropy_shannon_dec_11": 4.884,
    "tqwt_entropy_shannon_dec_9": 4.6877,
    "tqwt_energy_dec_33": 0.000002,
    "tqwt_energy_dec_31": 0.000007,
    "tqwt_energy_dec_28": 0.032743,
    "tqwt_energy_dec_27": 0.10807,
    "tqwt_energy_dec_26": 0.057575,
    "tqwt_energy_dec_25": 0.01007,
    "tqwt_energy_dec_16": 0.037555,
    "tqwt_energy_dec_14": 0.012066,
    "tqwt_energy_dec_12": 0.000239,
    "tqwt_energy_dec_7": 0.000164,
    "std_MFCC_8th_coef": 0.17101,
    "mean_MFCC_2nd_coef": 2.4874,
    "IMF_SNR_SEO": 51.6843,
    "VFER_mean": 0.000463,
    "f1": 539.342735,
    "DFA": 0.71826,
    "gender": 1.0,
}


class PredictRequest(BaseModel):
    features: dict[str, float]
    model: str = DEFAULT_MODEL

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "logistic_regression",
                "features": _EXAMPLE_FEATURES,
            }
        }
    }

    @model_validator(mode="after")
    def check_model_name(self):
        allowed = {"logistic_regression", "random_forest", "xgboost"}
        if self.model not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return self


class PredictResponse(BaseModel):
    model: str
    prediction: int
    label: str
    probability_parkinson: float
    probability_healthy: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "message": "Parkinson's Prediction API is running"}


@app.get("/features", tags=["Info"])
def list_features():
    """Return the list of feature names the model expects."""
    try:
        names = get_feature_names()
        return {"feature_count": len(names), "features": names}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict Parkinson's disease from voice features.

    Send a JSON body like:
    ```json
    {
      "features": {
        "PPE": 0.85,
        "DFA": 0.71,
        ...
      },
      "model": "logistic_regression"
    }
    ```
    """
    try:
        feature_names = get_feature_names()
        model = get_model(request.model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Build input vector; missing features default to 0.0
    missing = [f for f in feature_names if f not in request.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}. Send all {len(feature_names)} features.",
        )

    X = np.array([[request.features[f] for f in feature_names]])

    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        prob_healthy = float(proba[0])
        prob_parkinson = float(proba[1])
    else:
        score = float(model.decision_function(X)[0])
        prob_parkinson = round(1 / (1 + np.exp(-score)), 4)
        prob_healthy = round(1 - prob_parkinson, 4)

    return PredictResponse(
        model=request.model,
        prediction=pred,
        label="Parkinson" if pred == 1 else "Healthy",
        probability_parkinson=round(prob_parkinson, 4),
        probability_healthy=round(prob_healthy, 4),
    )


@app.get("/models", tags=["Info"])
def list_models():
    """List all available trained models."""
    if not os.path.exists(MODELS_DIR):
        return {"models": [], "message": "No models found. Run train.py first."}
    models = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    return {"models": models}
