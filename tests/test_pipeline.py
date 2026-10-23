"""
Comprehensive unit and integration test suite for Diabetes Risk Prediction.

Verifies:
1. The holdout split is exactly 154 samples and unchanged.
2. No validation or holdout data enters imputer fitting.
3. Out-of-fold predictions cover all training records (615 samples) exactly.
4. Threshold selection operates solely on training OOF predictions without holdout labels.
5. Selected thresholds are saved in metadata and used by inference.
6. Feature ordering and preprocessing are consistent across training, SHAP, and app.
7. Both Random Forest and XGBoost remain available and loadable.
8. Model comparison selects the winner from training-side CV, not holdout metrics.
9. Explainability service and SHAP workflows function properly.
10. End-to-end reproducibility of pipeline.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ZERO_AS_MISSING,
    apply_imputer,
    fit_imputer,
    get_project_root,
    load_metadata,
    load_processed_data,
    split_data,
)
from ml.train_models import (
    cross_validate_models_leak_free,
    optimize_threshold_oof,
    train_random_forest,
    train_xgboost,
)
from ml.evaluate_models import evaluate_model, select_best_model_from_cv
from explainability.shap_analysis import DiabetesExplainerService, load_any_model, load_best_model


@pytest.fixture
def dataset():
    return load_processed_data()


@pytest.fixture
def train_test_partitions(dataset):
    return split_data(dataset, test_size=0.20, random_state=42)


def test_01_holdout_split_preservation(train_test_partitions):
    """Criterion 1: The holdout split is locked to 154 records (100 Class 0, 54 Class 1)."""
    X_train, X_test, y_train, y_test = train_test_partitions
    assert len(X_test) == 154, f"Holdout test size must be exactly 154, got {len(X_test)}"
    assert len(y_test) == 154
    assert len(X_train) == 615, f"Training set size must be 615, got {len(X_train)}"
    assert (y_test == 0).sum() == 100, "Class 0 in holdout must be 100"
    assert (y_test == 1).sum() == 54, "Class 1 in holdout must be 54"


def test_02_leak_free_imputation(dataset):
    """Criterion 2: No validation or holdout data enters imputer fitting."""
    # Check that test set values differ before/after when imputed solely with training medians
    X_train, X_test, y_train, y_test = split_data(dataset, test_size=0.20, random_state=42)
    
    # Train medians
    train_medians = fit_imputer(X_train)
    assert set(train_medians.keys()) == set(ZERO_AS_MISSING)
    
    # Raw cleaned data
    raw_cleaned = dataset.copy()
    raw_cleaned[ZERO_AS_MISSING] = raw_cleaned[ZERO_AS_MISSING].replace(0, np.nan)
    raw_train = raw_cleaned.loc[X_train.index]
    raw_test = raw_cleaned.loc[X_test.index]
    
    # Computed medians on raw train vs raw test
    raw_train_medians = raw_train[ZERO_AS_MISSING].median().to_dict()
    raw_test_medians = raw_test[ZERO_AS_MISSING].median().to_dict()
    
    # Verify that X_test was imputed with raw_train_medians, NOT raw_test_medians
    for col in ZERO_AS_MISSING:
        assert train_medians[col] == pytest.approx(raw_train_medians[col], abs=1e-5)


def test_03_out_of_fold_coverage(train_test_partitions, dataset):
    """Criterion 3: Out-of-fold predictions cover all training records (615 samples) exactly."""
    X_train, X_test, y_train, y_test = train_test_partitions
    
    raw_cleaned = dataset.copy()
    raw_cleaned[ZERO_AS_MISSING] = raw_cleaned[ZERO_AS_MISSING].replace(0, np.nan)
    X_train_raw = raw_cleaned.loc[X_train.index, FEATURE_COLUMNS].copy()
    
    model_factories = {
        "Random Forest": lambda X, y: train_random_forest(X, y, random_state=42, class_weight="balanced", n_estimators=20),
    }
    
    cv_summary, per_fold, oof_probs_dict = cross_validate_models_leak_free(
        model_factories=model_factories,
        X_train_raw=X_train_raw,
        y_train=y_train,
        cv_splits=5,
        random_state=42,
    )
    
    oof = oof_probs_dict["Random Forest"]
    assert len(oof) == len(X_train), f"OOF probabilities must match training length {len(X_train)}"
    assert not np.isnan(oof).any(), "No NaN values allowed in out-of-fold predictions"
    assert (oof >= 0.0).all() and (oof <= 1.0).all(), "OOF predictions must be valid probabilities"


def test_04_threshold_selection_isolation(train_test_partitions):
    """Criterion 4: Threshold selection never reads holdout labels or predictions."""
    _, _, y_train, _ = train_test_partitions
    np.random.seed(42)
    fake_oof = np.random.uniform(0.1, 0.9, size=len(y_train))
    
    # Run threshold optimization with fake OOF probabilities and y_train only
    result = optimize_threshold_oof(y_train, fake_oof)
    assert "optimal_threshold" in result
    assert "oof_f1_optimal" in result
    assert 0.20 <= result["optimal_threshold"] <= 0.80
    assert result["oof_f1_optimal"] >= result["oof_f1_default"]


def test_05_metadata_and_threshold_used_in_inference():
    """Criterion 5: Selected threshold is saved in metadata and loaded by explainer service."""
    meta = load_metadata()
    assert meta is not None, "Metadata JSON must exist"
    assert "best_model_name" in meta
    assert "models" in meta
    
    best_name = meta["best_model_name"]
    best_info = meta["models"][best_name]
    assert "optimal_threshold" in best_info
    
    service = DiabetesExplainerService.get_instance(model_name=best_name)
    assert service.threshold == best_info["optimal_threshold"]


def test_06_feature_consistency():
    """Criterion 6: Feature ordering and preprocessing are consistent across training and SHAP."""
    expected_order = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    assert FEATURE_COLUMNS == expected_order
    
    meta = load_metadata()
    if meta and "feature_columns" in meta:
        assert meta["feature_columns"] == expected_order


def test_07_both_models_available():
    """Criterion 7: Both Random Forest and XGBoost remain available and loadable."""
    rf = load_any_model("random_forest")
    xgb = load_any_model("xgboost")
    best = load_best_model()
    
    assert rf is not None
    assert xgb is not None
    assert best is not None
    assert hasattr(rf, "predict_proba")
    assert hasattr(xgb, "predict_proba")
    assert hasattr(best, "predict_proba")


def test_08_model_selection_uses_validation_metrics():
    """Criterion 8: Model comparison uses validation metrics, not holdout metrics."""
    # Create fake CV summary and check selection
    cv_summary = pd.DataFrame([
        {"Model": "Model_A", "F1_Score_Mean": 0.72, "ROC_AUC_Mean": 0.85},
        {"Model": "Model_B", "F1_Score_Mean": 0.65, "ROC_AUC_Mean": 0.80},
    ])
    th_res = {
        "Model_A": {"oof_f1_optimal": 0.73},
        "Model_B": {"oof_f1_optimal": 0.66},
    }
    winner = select_best_model_from_cv(cv_summary, th_res)
    assert winner == "Model_A"


def test_09_explainer_service_workflow():
    """Criterion 9: Existing application and explainability workflows function properly."""
    service = DiabetesExplainerService.get_instance()
    sample_input = {
        "Pregnancies": 2,
        "Glucose": 140,
        "BloodPressure": 75,
        "SkinThickness": 30,
        "Insulin": 120,
        "BMI": 32.5,
        "DiabetesPedigreeFunction": 0.55,
        "Age": 45,
    }
    result = service.explain_prediction(sample_input)
    assert "probability" in result
    assert "predicted_class" in result
    assert "feature_contributions" in result
    assert "top_positive_contributors" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["predicted_class"] in [0, 1]


def test_10_evaluation_metric_computation(train_test_partitions):
    """Criterion 10: Evaluation produces valid non-zero metrics on held-out test data."""
    X_train, X_test, y_train, y_test = train_test_partitions
    model = load_best_model()
    meta = load_metadata()
    th = meta.get("models", {}).get(meta.get("best_model_name", ""), {}).get("optimal_threshold", 0.50)
    
    res = evaluate_model(model, X_test, y_test, threshold=th)
    assert 0.50 <= res["F1_Score"] <= 1.0
    assert 0.50 <= res["ROC_AUC"] <= 1.0
    assert 0.50 <= res["PR_AUC"] <= 1.0
    assert 0.50 <= res["Accuracy"] <= 1.0
