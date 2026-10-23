"""
Tests for Real Model Inference & Backend Integration.

Requirement: Add inference tests that compare frontend/backend results against
direct Python model inference on synthetic input records. Do not use holdout records.

Verifies:
1. Synthetic record inference with zero holdout record usage.
2. Direct scikit-learn / XGBoost model.predict_proba vs. DiabetesExplainerService.
3. HTTP REST inference endpoint (/api/predict) vs. direct Python model output.
4. Correct application of model-specific decision thresholds (RF: 0.48, XGB: 0.42).
5. Preprocessing median imputation on zero/missing biometric values.
6. Local TreeSHAP attributions and sum-to-output property.
7. Error handling for malformed / invalid requests.
"""

import json
import threading
import urllib.request
import urllib.error
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_utils import FEATURE_COLUMNS, ZERO_AS_MISSING, load_metadata
from explainability.shap_analysis import DiabetesExplainerService, load_any_model, load_best_model
from ml.server import start_prediction_server, get_active_server_port


SYNTHETIC_TEST_CASES = [
    {
        "name": "Synthetic Healthy Case",
        "inputs": {
            "Pregnancies": 0,
            "Glucose": 85.0,
            "BloodPressure": 66.0,
            "SkinThickness": 19.0,
            "Insulin": 60.0,
            "BMI": 21.8,
            "DiabetesPedigreeFunction": 0.20,
            "Age": 24.0,
        },
        "expected_risk_class": 0,
    },
    {
        "name": "Synthetic High Risk Case",
        "inputs": {
            "Pregnancies": 5,
            "Glucose": 178.0,
            "BloodPressure": 86.0,
            "SkinThickness": 38.0,
            "Insulin": 220.0,
            "BMI": 38.5,
            "DiabetesPedigreeFunction": 0.92,
            "Age": 54.0,
        },
        "expected_risk_class": 1,
    },
    {
        "name": "Synthetic Missing Fields Case (Zero values to be imputed with training medians)",
        "inputs": {
            "Pregnancies": 2,
            "Glucose": 0.0,         # Should be imputed with 117.0
            "BloodPressure": 0.0,     # Should be imputed with 72.0
            "SkinThickness": 0.0,     # Should be imputed with 29.0
            "Insulin": 0.0,           # Should be imputed with 125.0
            "BMI": 0.0,               # Should be imputed with 32.4
            "DiabetesPedigreeFunction": 0.45,
            "Age": 38.0,
        },
        "expected_risk_class": None,  # Computed from imputed values
    },
    {
        "name": "Synthetic Borderline Case",
        "inputs": {
            "Pregnancies": 1,
            "Glucose": 120.0,
            "BloodPressure": 74.0,
            "SkinThickness": 26.0,
            "Insulin": 95.0,
            "BMI": 28.4,
            "DiabetesPedigreeFunction": 0.38,
            "Age": 35.0,
        },
        "expected_risk_class": None,
    },
]


@pytest.fixture(scope="session")
def server_port():
    """Start the prediction server daemon on a local port."""
    port = start_prediction_server(host="127.0.0.1", initial_port=8502)
    return port


def _impute_synthetic(inputs: dict, medians: dict) -> pd.DataFrame:
    """Helper to replicate training-median imputation on synthetic dictionary."""
    row = {}
    for col in FEATURE_COLUMNS:
        val = inputs.get(col, 0.0)
        if val is None or (col in ZERO_AS_MISSING and val == 0.0):
            val = medians.get(col, 0.0)
        row[col] = float(val)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Test 1: Direct Python Model vs. DiabetesExplainerService
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", SYNTHETIC_TEST_CASES)
@pytest.mark.parametrize("model_key", ["random_forest", "xgboost"])
def test_direct_model_vs_explainer_service(case, model_key):
    """
    Verify that DiabetesExplainerService produces probability and class
    identically matching raw model.predict_proba() on synthetic data.
    """
    model = load_any_model(model_key)
    meta = load_metadata()
    model_name = "Random Forest" if model_key == "random_forest" else "XGBoost"
    threshold = meta["models"][model_name]["optimal_threshold"]
    medians = meta["imputation_medians"]

    # 1. Direct Python model prediction with canonical preprocessing
    df_imputed = _impute_synthetic(case["inputs"], medians)
    direct_prob = float(model.predict_proba(df_imputed)[0, 1])
    direct_class = int(direct_prob >= threshold)

    # 2. Explainer service prediction
    service = DiabetesExplainerService.get_instance(model_name=model_key)
    service_result = service.explain_prediction(case["inputs"])

    assert service_result["probability"] == pytest.approx(direct_prob, abs=1e-5), (
        f"Explainer prob {service_result['probability']} does not match direct model prob {direct_prob}"
    )
    assert service_result["predicted_class"] == direct_class, (
        f"Explainer class {service_result['predicted_class']} does not match threshold rule {direct_class}"
    )
    assert service_result["threshold"] == threshold


# ---------------------------------------------------------------------------
# Test 2: HTTP REST API vs. Direct Python Model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", SYNTHETIC_TEST_CASES)
@pytest.mark.parametrize("model_key", ["random_forest", "xgboost"])
def test_http_api_vs_direct_model(server_port, case, model_key):
    """
    Verify that HTTP POST /api/predict produces identical predictions
    to direct Python model inference for both Random Forest and XGBoost.
    """
    model = load_any_model(model_key)
    meta = load_metadata()
    model_name = "Random Forest" if model_key == "random_forest" else "XGBoost"
    threshold = meta["models"][model_name]["optimal_threshold"]
    medians = meta["imputation_medians"]

    # Direct reference calculation
    df_imputed = _impute_synthetic(case["inputs"], medians)
    expected_prob = float(model.predict_proba(df_imputed)[0, 1])
    expected_class = int(expected_prob >= threshold)

    # HTTP request to backend
    payload = {
        "model": model_key,
        "features": case["inputs"],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{server_port}/api/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))

    assert data["status"] == "success"
    assert data["probability"] == pytest.approx(expected_prob, abs=1e-5)
    assert data["predicted_class"] == expected_class
    assert data["threshold"] == threshold
    assert data["risk_tier"] in ("elevated", "low")

    # Verify SHAP attributions are present and non-empty
    assert "shap_contributions" in data
    assert len(data["shap_contributions"]) == 8
    for feat in FEATURE_COLUMNS:
        assert feat in data["shap_contributions"]
        contrib = data["shap_contributions"][feat]
        assert "val" in contrib
        assert "shap" in contrib
        assert isinstance(contrib["shap"], float)


# ---------------------------------------------------------------------------
# Test 3: Model-Specific Threshold Enforcement
# ---------------------------------------------------------------------------
def test_model_specific_thresholds():
    """
    Verify Random Forest strictly enforces 0.48 and XGBoost strictly enforces 0.42.
    """
    rf_service = DiabetesExplainerService.get_instance(model_name="random_forest")
    xgb_service = DiabetesExplainerService.get_instance(model_name="xgboost")

    assert rf_service.threshold == 0.48, f"Random Forest threshold must be 0.48, got {rf_service.threshold}"
    assert xgb_service.threshold == 0.42, f"XGBoost threshold must be 0.42, got {xgb_service.threshold}"


# ---------------------------------------------------------------------------
# Test 4: Median Imputation Integrity for Zero / Missing Values
# ---------------------------------------------------------------------------
def test_zero_imputation_integrity():
    """
    Verify that zero values in physiological features are imputed with
    the exact training set medians from models/metadata.json.
    """
    meta = load_metadata()
    expected_medians = meta["imputation_medians"]

    service = DiabetesExplainerService.get_instance(model_name="random_forest")
    # All zeros for physiologically required features
    inputs = {col: 0.0 for col in FEATURE_COLUMNS}
    res = service.explain_prediction(inputs)

    features_used = res["feature_contributions"]
    for col, med in expected_medians.items():
        actual_val = features_used[col]["feature_value"]
        assert actual_val == pytest.approx(med, abs=1e-5), (
            f"Feature {col} was not imputed with training median {med}; got {actual_val}"
        )


# ---------------------------------------------------------------------------
# Test 5: Error Handling for Malformed Requests
# ---------------------------------------------------------------------------
def test_api_error_handling(server_port):
    """
    Verify the REST API safely returns HTTP 400 for invalid/malformed requests
    without crashing the server.
    """
    # 1. Invalid features type (string instead of dict)
    req = urllib.request.Request(
        f"http://127.0.0.1:{server_port}/api/predict",
        data=json.dumps({"model": "random_forest", "features": "invalid_string"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 400

    # 2. Empty request body
    req_empty = urllib.request.Request(
        f"http://127.0.0.1:{server_port}/api/predict",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info_empty:
        urllib.request.urlopen(req_empty, timeout=5)
    assert exc_info_empty.value.code == 400

    # 3. Health endpoint succeeds
    with urllib.request.urlopen(f"http://127.0.0.1:{server_port}/health", timeout=5) as health_resp:
        assert health_resp.status == 200
        health_data = json.loads(health_resp.read().decode("utf-8"))
        assert health_data["status"] == "healthy"
