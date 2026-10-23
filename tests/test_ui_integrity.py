"""
Automated UI and Data Integrity Tests for Diabetes Risk Prediction.

Phase 7: Verifies that every displayed metric, threshold, clinical scenario,
and feature importance matches the authoritative artifacts and backend contracts.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def frontend_html() -> str:
    path = PROJECT_ROOT / "frontend" / "index.html"
    assert path.exists(), "frontend/index.html does not exist"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def metadata() -> dict:
    path = PROJECT_ROOT / "models" / "metadata.json"
    assert path.exists(), "models/metadata.json does not exist"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def predictions_df() -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "predictions.csv"
    assert path.exists(), "results/predictions.csv does not exist"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def metrics_df() -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "metrics.csv"
    assert path.exists(), "results/metrics.csv does not exist"
    return pd.read_csv(path)


def test_01_benchmark_metrics_match_predictions(predictions_df, metadata, metrics_df):
    """Verify that metrics recalculated from predictions.csv match metadata and metrics.csv."""
    assert len(predictions_df) == 154, "Holdout set must contain exactly 154 records"
    y_true = predictions_df["Actual"].values

    # 1. Random Forest
    rf_prob = predictions_df["Random_Forest_Probability"].values
    rf_pred = predictions_df["Random_Forest_Prediction"].values
    rf_th = metadata["models"]["Random Forest"]["optimal_threshold"]
    assert rf_th == 0.48

    # Ensure class predictions match (prob >= threshold)
    np.testing.assert_array_equal(rf_pred, (rf_prob >= rf_th).astype(int))

    rf_acc = accuracy_score(y_true, rf_pred)
    rf_prec = precision_score(y_true, rf_pred)
    rf_rec = recall_score(y_true, rf_pred)
    rf_f1 = f1_score(y_true, rf_pred)
    rf_roc = roc_auc_score(y_true, rf_prob)

    assert round(rf_acc * 100, 2) == 73.38
    assert round(rf_prec * 100, 2) == 58.67
    assert round(rf_rec * 100, 2) == 81.48
    assert round(rf_f1 * 100, 2) == 68.22
    assert round(rf_roc, 4) == 0.8333

    # 2. XGBoost
    xgb_prob = predictions_df["XGBoost_Probability"].values
    xgb_pred = predictions_df["XGBoost_Prediction"].values
    xgb_th = metadata["models"]["XGBoost"]["optimal_threshold"]
    assert xgb_th == 0.42

    np.testing.assert_array_equal(xgb_pred, (xgb_prob >= xgb_th).astype(int))

    xgb_acc = accuracy_score(y_true, xgb_pred)
    xgb_prec = precision_score(y_true, xgb_pred)
    xgb_rec = recall_score(y_true, xgb_pred)
    xgb_f1 = f1_score(y_true, xgb_pred)
    xgb_roc = roc_auc_score(y_true, xgb_prob)

    assert round(xgb_acc * 100, 2) == 74.03
    assert round(xgb_prec * 100, 2) == 58.97
    assert round(xgb_rec * 100, 2) == 85.19
    assert round(xgb_f1 * 100, 2) == 69.70
    assert round(xgb_roc, 4) == 0.8404


def test_02_ui_table_values_match_verified_metrics(frontend_html):
    """Verify that benchmark table in frontend HTML contains the exact verified numbers."""
    # Random Forest Row
    assert "73.38%" in frontend_html
    assert "58.67%" in frontend_html
    assert "81.48%" in frontend_html
    assert "68.22%" in frontend_html
    assert "0.8333" in frontend_html
    assert "Th: 0.48" in frontend_html

    # XGBoost Row
    assert "74.03%" in frontend_html
    assert "58.97%" in frontend_html
    assert "85.19%" in frontend_html
    assert "69.70%" in frontend_html
    assert "0.8404" in frontend_html
    assert "Th: 0.42" in frontend_html


def test_03_model_specification_strip_and_kpi_cards(frontend_html, metadata):
    """Verify technical specification strip and KPI cards in frontend HTML."""
    rf_trees = metadata["models"]["Random Forest"]["params"]["n_estimators"]
    xgb_trees = metadata["models"]["XGBoost"]["params"]["n_estimators"]

    assert f"Random Forest (Selected Best • n={rf_trees})" in frontend_html
    assert f"XGBoost (Tuned • n={xgb_trees})" in frontend_html
    assert "Pima Indians Diabetes (N=769, 8 Features, Holdout N=154)" in frontend_html

    # Default KPI card matches Random Forest
    assert 'id="kpiModelName">Random Forest (Selected Best)</span>' in frontend_html
    assert 'id="kpiModelAuc">Holdout AUC: 0.8333</span>' in frontend_html
    assert 'id="kpiModelF1">• Holdout F1: 68.22% (Th: 0.48)</span>' in frontend_html


def test_04_feature_importance_values_match_models(frontend_html):
    """Verify that RF Gini importance bars in JS match random_forest.joblib."""
    import joblib
    rf = joblib.load(PROJECT_ROOT / "models" / "random_forest.joblib")
    importances = rf.feature_importances_
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    imp_dict = dict(zip(cols, importances))

    # Glucose is top predictor (~0.341)
    assert round(imp_dict["Glucose"], 3) == 0.341
    assert "0.341" in frontend_html
    # BMI is second (~0.185)
    assert round(imp_dict["BMI"], 3) == 0.185
    assert "0.185" in frontend_html
    # Age is third (~0.130)
    assert round(imp_dict["Age"], 3) == 0.130
    assert "0.130" in frontend_html
    # Insulin is fourth (~0.094)
    assert round(imp_dict["Insulin"], 3) == 0.094
    assert "0.094" in frontend_html


def test_05_clinical_test_cases_integrity(frontend_html):
    """Verify that clinical test cases TEST 01, 02, 03 are present with non-fabricated labels."""
    assert "TEST 01" in frontend_html
    assert "TEST 02" in frontend_html
    assert "TEST 03" in frontend_html

    assert "Elevated Glycemia" in frontend_html
    assert "Normoglycemic Baseline" in frontend_html
    assert "Impaired Fasting Glucose" in frontend_html

    # Values in TEST_CASES JS dictionary
    assert "test-01" in frontend_html
    assert "gluc: 154" in frontend_html
    assert "test-02" in frontend_html
    assert "gluc: 88" in frontend_html
    assert "test-03" in frontend_html
    assert "gluc: 126" in frontend_html


def test_06_error_handling_no_fabricated_predictions(frontend_html):
    """Verify that frontend API error handler does not fabricate predictions."""
    assert "Inference Service Unreachable" in frontend_html
    assert "Backend Error:" in frontend_html
    assert "calcRisk" in frontend_html or "calculateRisk" in frontend_html


def test_07_default_unselected_clinical_test_cases(frontend_html):
    """Verify that all clinical test-case cards start neutral/unselected and inputs start blank."""
    import re

    # 1. activeTestCase initialized to null
    assert "let activeTestCase = null;" in frontend_html

    # 2. All 3 cards start with aria-pressed="false"
    assert re.search(r'id="testCard-01"[^>]*aria-pressed="false"', frontend_html) is not None
    assert re.search(r'id="testCard-02"[^>]*aria-pressed="false"', frontend_html) is not None
    assert re.search(r'id="testCard-03"[^>]*aria-pressed="false"', frontend_html) is not None

    # 3. Card 01 does not have hardcoded active indicator on initial load
    card_01_match = re.search(r'<div id="testCard-01".*?</div>\s*</div>\s*</div>', frontend_html, re.DOTALL)
    assert card_01_match is not None
    card_01_html = card_01_match.group(0)
    assert "radio_button_unchecked" in card_01_html
    assert "test-card-status-text\">Select</span>" in card_01_html
    assert "border-primary bg-primary/5" not in card_01_html

    # 4. Header scenario indicator tag removed as requested
    assert 'id="currentCohortLabel"' not in frontend_html

    # 5. Form inputs initialize to empty strings, not prefilled Test 1 values
    for field_id in ["inputAge", "inputPregnancies", "inputGlucose", "inputBP", "inputBMI", "inputInsulin", "inputSkin", "inputPedigree"]:
        pattern = rf'id="{field_id}"[^>]*value=""'
        assert re.search(pattern, frontend_html) is not None, f"Field {field_id} should start with empty value"

    # 6. Glucose badge starts hidden
    assert '<span class="hidden" id="glucoseBadge">' in frontend_html

    # 7. JavaScript input listeners handle clearing active test case on manual edits
    assert "handleFormInputChange" in frontend_html
    assert "activeTestCase = null;" in frontend_html
    assert "updateTestCaseCardsUI(null)" in frontend_html

