"""
End-to-End Execution Script for Diabetes Risk Prediction ML Pipeline.

Executes:
1. Data loading and schema validation
2. Stratified train/test split without test leakage
3. Controlled hyperparameter search for Random Forest and XGBoost
4. 5-Fold Stratified Cross-Validation on training data
5. Single-pass final evaluation on untouched test set
6. Publication-grade visualization (Confusion Matrices, ROC Curve, Feature Importances, Metrics)
7. Dynamic model selection & artifact persistence
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_project_root,
    load_processed_data,
    split_data,
)
from ml.train_models import (
    cross_validate_models,
    train_random_forest,
    train_xgboost,
    tune_random_forest,
    tune_xgboost,
)
from ml.evaluate_models import evaluate_and_save_pipeline


def main() -> None:
    print("=" * 70)
    print("EXPLAINABLE DIABETES RISK PREDICTION - ML PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/6] Loading and validating dataset...")
    data = load_processed_data()
    print(f"  Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"  Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"  Target: {TARGET_COLUMN}")
    target_counts = data[TARGET_COLUMN].value_counts().to_dict()
    print(f"  Target distribution: Class 0 (Non-Diabetic) = {target_counts.get(0, 0)}, Class 1 (Diabetic) = {target_counts.get(1, 0)}")

    # 2. Stratified Train-Test Split (leak-free)
    print("\n[Step 2/6] Performing stratified 80/20 train-test split (random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.20, random_state=42)
    print(f"  Training set: {X_train.shape[0]} samples (Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()})")
    print(f"  Testing set:  {X_test.shape[0]} samples (Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()})")
    print("  Verification: Test data medians computed solely from training set (Zero Data Leakage).")

    # 3. Model Training & Hyperparameter Tuning
    print("\n[Step 3/6] Controlled hyperparameter tuning on training set (5-Fold Stratified CV, F1-score)...")

    print("  --> Tuning Random Forest...")
    t0 = time.time()
    best_rf, rf_best_params, rf_best_score = tune_random_forest(X_train, y_train, random_state=42)
    print(f"      Completed in {time.time() - t0:.2f}s | Best Training F1: {rf_best_score:.4f}")
    print(f"      Best Hyperparameters: {rf_best_params}")

    print("  --> Tuning XGBoost...")
    t0 = time.time()
    best_xgb, xgb_best_params, xgb_best_score = tune_xgboost(X_train, y_train, random_state=42)
    print(f"      Completed in {time.time() - t0:.2f}s | Best Training F1: {xgb_best_score:.4f}")
    print(f"      Best Hyperparameters: {xgb_best_params}")

    selected_models = {
        "Random Forest": best_rf,
        "XGBoost": best_xgb,
    }

    # 4. 5-Fold Stratified Cross-Validation on Training Data
    print("\n[Step 4/6] Conducting 5-Fold Stratified Cross-Validation across candidate models...")
    cv_results_df = cross_validate_models(selected_models, X_train, y_train, cv_splits=5, random_state=42)
    print("\nCross-Validation Performance (Training Folds):")
    print(cv_results_df.to_string(index=False))

    # 5. Final Evaluation on Untouched Test Set
    print("\n[Step 5/6] Final single-pass evaluation on untouched test set...")
    output_dirs = [
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "ml" / "results",
    ]
    results = evaluate_and_save_pipeline(
        selected_models=selected_models,
        cv_results_df=cv_results_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dirs=output_dirs,
    )

    print("\nFinal Test Set Performance Comparison:")
    print(results["comparison_df"].to_string(index=False))

    print(f"\n[Step 6/6] Model Selection & Result Persistence:")
    print(f"  Best-Performing Model: >>> {results['best_model_name']} <<<")
    print("  Saved artifacts:")
    print("   - models/best_model.pkl & models/best_model.joblib")
    print("   - models/random_forest.pkl & models/xgboost.pkl")
    print("   - results/metrics.csv")
    print("   - results/cross_validation_results.csv")
    print("   - results/model_comparison.csv")
    print("   - results/predictions.csv")
    print("   - results/confusion_matrix_random_forest.png")
    print("   - results/confusion_matrix_xgboost.png")
    print("   - results/roc_curve_comparison.png")
    print("   - results/metric_comparison.png")
    print("   - results/random_forest_feature_importance.png")
    print("   - results/xgboost_feature_importance.png")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
