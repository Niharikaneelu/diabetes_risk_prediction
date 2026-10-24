"""
Tests for Two-Page Architecture, Four-Stage Inference Transition,
State Management, Input Invalidation, and Results View Integrity.
"""

import json
import re
from pathlib import Path
import pytest
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="module")
def frontend_html() -> str:
    path = PROJECT_ROOT / "frontend" / "index.html"
    assert path.exists(), "frontend/index.html does not exist"
    return path.read_text(encoding="utf-8")


def test_01_two_distinct_views_exist(frontend_html: str):
    """Verify that Page 1 (Risk Assessment) and Page 2 (Prediction Results) exist as separate views."""
    # Page 1: Risk Assessment container
    assert 'id="tab-content-risk-prediction"' in frontend_html
    # Page 2: Dedicated Prediction Results container
    assert 'id="tab-content-prediction-results"' in frontend_html

    # Navigation buttons for both pages
    assert 'id="nav-risk-prediction"' in frontend_html
    assert 'id="nav-prediction-results"' in frontend_html
    assert "Risk Assessment" in frontend_html
    assert "Prediction Results" in frontend_html


def test_02_page_1_assessment_content_and_results_exclusion(frontend_html: str):
    """Verify Page 1 contains form, clinical test cases, model selector, but NO inline results."""
    # Split Page 1 content
    p1_start = frontend_html.find('id="tab-content-risk-prediction"')
    assert p1_start != -1

    # End of Page 1 is before the inference modal
    p1_end = frontend_html.find('id="inferenceTransitionModal"')
    assert p1_end != -1
    page1_content = frontend_html[p1_start:p1_end]

    # Required items on Page 1
    assert 'id="predictionForm"' in page1_content
    assert 'id="btnCalculate"' in page1_content
    assert "Estimate Diabetes Risk" in page1_content
    assert "Clear Fields" in page1_content
    assert "resetForm()" in page1_content

    # Clinical test cases
    assert "TEST 01" in page1_content
    assert "TEST 02" in page1_content
    assert "TEST 03" in page1_content
    assert "Elevated Glycemia" in page1_content
    assert "Normoglycemic Baseline" in page1_content
    assert "Impaired Fasting Glucose" in page1_content

    # Model selector banner strip
    assert 'id="bannerModelTitle"' in page1_content
    assert 'id="toggleBtnRf"' in page1_content
    assert 'id="toggleBtnXgb"' in page1_content

    # All 8 input fields
    for field in ["inputGlucose", "inputBMI", "inputAge", "inputBP", "inputPregnancies", "inputInsulin", "inputSkin", "inputPedigree"]:
        assert f'id="{field}"' in page1_content

    # Assessment error banner for retry action
    assert 'id="assessmentErrorBanner"' in page1_content
    assert "Retry Inference" in page1_content

    # Ensure prediction gauge and SHAP waterfall are NOT in Page 1
    assert 'id="riskGaugeValue"' not in page1_content
    assert 'id="riskGaugeCircle"' not in page1_content
    assert 'id="shap-analysis-section"' not in page1_content
    assert 'id="divergenceRows"' not in page1_content


def test_03_inference_transition_experience(frontend_html: str):
    """Verify minimal, clean evaluation screen with waveform icon, Inference Engine label, and heading."""
    assert 'id="inferenceTransitionModal"' in frontend_html
    assert "Evaluating Diabetes Risk" in frontend_html
    assert "vital_signs" in frontend_html
    assert "Inference Engine" in frontend_html
    assert "eval-pulse-glow" in frontend_html

    # Verify progress bar, 4 stages, and footer tags are completely removed
    assert 'id="inferenceProgressBar"' not in frontend_html
    assert 'id="inferenceStagesContainer"' not in frontend_html
    assert 'id="stageItem1"' not in frontend_html
    assert 'id="stageItem2"' not in frontend_html
    assert 'id="stageItem3"' not in frontend_html
    assert 'id="stageItem4"' not in frontend_html
    assert "1. Biometric Ingestion" not in frontend_html
    assert "2. Ensemble Model Execution" not in frontend_html
    assert "3. TreeSHAP Feature Attribution" not in frontend_html
    assert "4. Risk Calibration" not in frontend_html
    assert 'id="inferenceModelTag"' not in frontend_html


def test_04_page_2_prediction_results_structure(frontend_html: str):
    """Verify Page 2 contains dedicated results view, gauge, SHAP, patient inputs, disclaimer, and actions."""
    p2_start = frontend_html.find('id="tab-content-prediction-results"')
    assert p2_start != -1

    # End of Page 2 is before Tab 2 (explainability-xai)
    p2_end = frontend_html.find('id="tab-content-explainability-xai"')
    assert p2_end != -1
    page2_content = frontend_html[p2_start:p2_end]

    # Navigation actions
    assert "Back to Assessment" in page2_content
    assert "Export Report" in page2_content
    assert "switchTab('risk-prediction')" in page2_content

    # Graceful empty state without fabricated data
    assert 'id="resultsEmptyState"' in page2_content
    assert "No Assessment Results Available" in page2_content
    assert 'id="resultsContent"' in page2_content

    # Animated probability gauge
    assert 'id="riskGaugeCircle"' in page2_content
    assert 'id="riskGaugeValue"' in page2_content
    assert 'id="riskGaugeThreshold"' in page2_content

    # Predicted class and qualified risk interpretation
    assert 'id="riskPill"' in page2_content
    assert 'id="riskPillText"' in page2_content
    assert 'id="riskPhenotypeDesc"' in page2_content
    assert 'id="spectrumMarker"' in page2_content
    assert 'id="confidenceValue"' in page2_content

    # Actual model used
    assert 'id="resultModelName"' in page2_content
    assert 'id="resultThresholdBadge"' in page2_content

    # Local SHAP explanation
    assert 'id="shap-analysis-section"' in page2_content
    assert 'id="divergenceRows"' in page2_content
    assert 'id="increasingFactorsList"' in page2_content
    assert 'id="decreasingFactorsList"' in page2_content
    assert 'id="explainerModelBadge"' in page2_content
    assert 'id="waterfallFormulaText"' in page2_content
    assert 'id="waterfallBaseFormula"' in page2_content
    assert 'id="waterfallPredictedFormula"' in page2_content

    # Concise patient input summary card
    assert 'id="patientInputSummaryCard"' in page2_content
    assert 'id="patientInputSummaryTiles"' in page2_content
    assert "Patient Inputs Used for this Assessment" in page2_content

    # Clinical disclaimer
    assert "Clinical Decision Support Notice &amp; Disclaimer" in page2_content

    # Global feature importance and model specs
    assert 'id="importanceBarsContainer"' in page2_content
    assert 'id="btnImpRf"' in page2_content
    assert 'id="btnImpXgb"' in page2_content


def test_05_state_management_and_invalidation_logic(frontend_html: str):
    """Verify state management, stale result invalidation, and race-condition prevention in JS."""
    # State record variable
    assert "currentResultRecord" in frontend_html
    # Invalidation function
    assert "invalidateStaleResults" in frontend_html
    # Duplicate request prevention
    assert "isCalculating" in frontend_html
    # Late response prevention
    assert "activeRequestId" in frontend_html

    # Form input listeners for invalidation
    assert "invalidateStaleResults('inputs_changed')" in frontend_html
    assert "invalidateStaleResults('model_changed')" in frontend_html

    # Popstate routing listener
    assert "window.addEventListener('popstate'" in frontend_html


@pytest.fixture(scope="module")
def api_server_port():
    from ml.server import start_prediction_server, get_active_server_port
    port = start_prediction_server(host="127.0.0.1", initial_port=8502)
    return port


def test_06_e2e_backend_inference_and_record_contract(api_server_port):
    """Verify /api/predict provides full inference & SHAP attributes required for Page 2 record."""
    url = f"http://127.0.0.1:{api_server_port}/api/predict"

    for model_name in ["random_forest", "xgboost"]:
        payload = {
            "model": model_name,
            "features": {
                "Pregnancies": 2,
                "Glucose": 154.0,
                "BloodPressure": 78.0,
                "SkinThickness": 32.0,
                "Insulin": 130.0,
                "BMI": 31.4,
                "DiabetesPedigreeFunction": 0.672,
                "Age": 48
            }
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 200
            res = json.loads(response.read().decode("utf-8"))

            assert res["status"] == "success"
            assert "probability" in res
            assert 0.0 <= res["probability"] <= 1.0
            assert "threshold" in res
            assert "predicted_class" in res
            assert "shap_contributions" in res
            assert len(res["shap_contributions"]) == 8
            assert "base_value" in res
            assert "risk_tier" in res
            assert "model_name" in res


def test_07_landing_page_integrity(frontend_html: str):
    """Verify landing page exists with exact specified text and navigation button."""
    assert 'id="tab-content-landing"' in frontend_html
    assert 'id="btnExploreApp"' in frontend_html
    assert "Explainable Diabetes Risk Prediction using Machine Learning" in frontend_html
    assert "A Comparative Analysis of Random Forest and XGBoost with SHAP" in frontend_html
    assert "switchTab('risk-prediction')" in frontend_html

