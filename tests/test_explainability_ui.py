import math
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.shap_analysis import DiabetesExplainerService

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def test_explainability_mathematical_consistency():
    """Verify SHAP additive decomposition and probability mapping for both models on synthetic cases."""
    rf_service = DiabetesExplainerService(model_name="random_forest")
    xgb_service = DiabetesExplainerService(model_name="xgboost")

    test_biometrics = {
        "Pregnancies": 1,
        "Glucose": 115.0,
        "BloodPressure": 70.0,
        "SkinThickness": 22.0,
        "Insulin": 85.0,
        "BMI": 26.5,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 35
    }

    # 1. Random Forest (Probability Space)
    rf_res = rf_service.explain_prediction(test_biometrics)
    rf_base = rf_res["base_value"]
    rf_shap_sum = sum(c["shap_value"] for c in rf_res["feature_contributions"].values())
    rf_pred_prob = rf_res["probability"]
    assert rf_res["output_space"] == "probability"
    assert abs(rf_base + rf_shap_sum - rf_pred_prob) < 1e-4, f"RF Additive inconsistency: {rf_base} + {rf_shap_sum} != {rf_pred_prob}"

    # 2. XGBoost (Log-Odds Margin Space -> Sigmoid)
    xgb_res = xgb_service.explain_prediction(test_biometrics)
    xgb_base = xgb_res["base_value"]
    xgb_shap_sum = sum(c["shap_value"] for c in xgb_res["feature_contributions"].values())
    xgb_margin = xgb_res["margin_output"]
    xgb_pred_prob = xgb_res["probability"]
    assert xgb_res["output_space"] == "log_odds"
    assert abs(xgb_base + xgb_shap_sum - xgb_margin) < 1e-4, f"XGB Margin additive inconsistency: {xgb_base} + {xgb_shap_sum} != {xgb_margin}"
    assert abs(sigmoid(xgb_margin) - xgb_pred_prob) < 1e-4, f"XGB Sigmoid inconsistency: sigmoid({xgb_margin}) != {xgb_pred_prob}"

def test_explainability_ui_structure():
    """Verify frontend HTML contains consolidated explainability sections and no redundant duplicates."""
    html_path = Path("frontend/index.html")
    assert html_path.exists(), "frontend/index.html must exist"
    content = html_path.read_text(encoding="utf-8")

    # Required Section Headers
    assert "Why did the model make this prediction?" in content
    assert "Global Feature Importance" in content
    assert "Which features influence the model most across the evaluated dataset?" in content

    # Retained Key Elements
    assert 'id="shap-analysis-section"' in content
    assert 'id="divergenceRows"' in content
    assert 'id="increasingFactorsList"' in content
    assert 'id="decreasingFactorsList"' in content
    assert 'id="importanceBarsContainer"' in content
    assert 'id="btnImpRf"' in content
    assert 'id="btnImpXgb"' in content
    assert 'id="explainerModelBadge"' in content

    # Removed Redundant Components
    # Stepper container must be removed from rendered DOM
    assert 'id="stepperContainer"' not in content
    # Card A and Card B large duplicate panels must not exist as separate full sections
    assert "Card A: Risk Factors" not in content
    assert "Card B: Protective Factors" not in content
