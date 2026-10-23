"""
End-to-End Execution Script for Leakage-Safe Diabetes Risk Prediction ML Pipeline.

Executes:
1. Data loading and schema validation
2. Stratified 80/20 train/test split with strict holdout isolation
3. Leak-free Stratified 5-Fold Cross-Validation on training data
4. Out-of-fold probability generation and F1-maximizing threshold optimization
5. Leakage-free model selection based strictly on training-side validation metrics
6. Freezing of all decisions (models, parameters, medians, thresholds)
7. Single-pass final evaluation on the untouched 154-record holdout test set
8. Publication-grade visualization (Confusion Matrices, ROC, PR Curve, Feature Importances, Metrics)
9. Artifact and structured metadata persistence (JSON + joblib + pickle)
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
    ZERO_AS_MISSING,
    fit_imputer,
    get_project_root,
    load_processed_data,
    split_data,
)
from ml.train_models import (
    cross_validate_models_leak_free,
    optimize_threshold_oof,
    train_random_forest,
    train_xgboost,
)
from ml.evaluate_models import evaluate_and_save_pipeline, select_best_model_from_cv


def main() -> None:
    print("=" * 75)
    print("LEAKAGE-SAFE EXPLAINABLE DIABETES RISK PREDICTION PIPELINE")
    print("=" * 75)

    # 1. Load Data
    print("\n[Step 1/6] Loading and validating dataset...")
    data = load_processed_data()
    print(f"  Total records: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"  Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"  Target: {TARGET_COLUMN}")
    target_counts = data[TARGET_COLUMN].value_counts().to_dict()
    print(f"  Target distribution: Class 0 (Non-Diabetic) = {target_counts.get(0, 0)}, Class 1 (Diabetic) = {target_counts.get(1, 0)}")

    # 2. Stratified Train-Test Split (strict holdout preservation)
    print("\n[Step 2/6] Performing stratified 80/20 train-test split (random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.20, random_state=42)
    print(f"  Training partition: {X_train.shape[0]} samples (Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()})")
    print(f"  Holdout test set:   {X_test.shape[0]} samples (Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()})")
    print("  [LOCK] Holdout test set locked. It will NOT be accessed during CV, tuning, or selection.")

    # Calculate full-training-set imputation medians (to be saved with model metadata)
    train_imputation_medians = fit_imputer(X_train)
    print(f"  Training Imputation Medians: {train_imputation_medians}")

    # Raw training data with NAs for fold-level cross-validation
    raw_cleaned = data.copy()
    raw_cleaned[ZERO_AS_MISSING] = raw_cleaned[ZERO_AS_MISSING].replace(0, None)
    for col in ZERO_AS_MISSING:
        raw_cleaned[col] = raw_cleaned[col].astype(float)
    X_train_raw = raw_cleaned.loc[X_train.index, FEATURE_COLUMNS].copy()

    # 3. Model Definition & Leak-Free 5-Fold Stratified Cross-Validation
    print("\n[Step 3/6] Running leak-free 5-Fold Stratified Cross-Validation on training data...")
    print("  (Imputation medians fitted strictly per training fold; zero validation contamination)")

    model_factories = {
        "Random Forest": lambda X, y: train_random_forest(X, y, random_state=42, class_weight="balanced"),
        "XGBoost": lambda X, y: train_xgboost(X, y, random_state=42),
    }

    t0 = time.time()
    cv_summary_df, per_fold_dict, oof_probs_dict = cross_validate_models_leak_free(
        model_factories=model_factories,
        X_train_raw=X_train_raw,
        y_train=y_train,
        cv_splits=5,
        random_state=42,
    )
    print(f"  CV completed in {time.time() - t0:.2f}s")
    print("\nTraining-Side 5-Fold Stratified Cross-Validation Performance (th=0.50):")
    print(cv_summary_df.to_string(index=False))

    # 4. Out-of-Fold Threshold Optimization
    print("\n[Step 4/6] Optimizing classification thresholds on Out-Of-Fold training probabilities...")
    threshold_results = {}
    for name, oof_probs in oof_probs_dict.items():
        th_opt = optimize_threshold_oof(y_train, oof_probs, threshold_min=0.20, threshold_max=0.80, threshold_step=0.02)
        threshold_results[name] = th_opt
        print(f"  --> {name}:")
        print(f"      Default (th=0.50): OOF F1 = {th_opt['oof_f1_default']:.4f} | Recall = {th_opt['oof_recall_default']:.4f} | Precision = {th_opt['oof_precision_default']:.4f}")
        print(f"      Optimal (th={th_opt['optimal_threshold']:.2f}): OOF F1 = {th_opt['oof_f1_optimal']:.4f} | Recall = {th_opt['oof_recall_optimal']:.4f} | Precision = {th_opt['oof_precision_optimal']:.4f}")
        print(f"      F1 Improvement Gain: {th_opt['f1_gain']:+.4f}")

    # 5. Leakage-Free Model Selection (Training Metrics Only)
    print("\n[Step 5/6] Leakage-Free Model Selection (Prioritizing Training F1 & ROC-AUC)...")
    best_model_name = select_best_model_from_cv(cv_summary_df, threshold_results)
    print(f"  Identified Best Architecture from Training CV: >>> {best_model_name} <<<")
    print("  [FREEZE] Freezing all model hyper-parameters, fitted medians, and decision thresholds.")

    # Instantiate full models for final training
    selected_models = {
        "Random Forest": train_random_forest(X_train, y_train, random_state=42, class_weight="balanced"),
        "XGBoost": train_xgboost(X_train, y_train, random_state=42),
    }

    # 6. Single-Pass Final Evaluation on Untouched Holdout Test Set
    print("\n[Step 6/6] Executing single-pass evaluation on untouched holdout test set...")
    output_dirs = [
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "ml" / "results",
    ]

    results = evaluate_and_save_pipeline(
        selected_models=selected_models,
        best_model_name=best_model_name,
        cv_summary_df=cv_summary_df,
        threshold_results=threshold_results,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        imputation_medians=train_imputation_medians,
        output_dirs=output_dirs,
        random_state=42,
    )

    print("\nFinal Holdout Test Set Performance Comparison:")
    print(results["comparison_df"].to_string(index=False))

    print("\nSaved Artifacts:")
    print("  - models/best_model.pkl & models/best_model.joblib")
    print("  - models/random_forest.pkl & models/random_forest.joblib")
    print("  - models/xgboost.pkl & models/xgboost.joblib")
    print("  - models/metadata.json (complete audit trail & thresholds)")
    print("  - results/metrics.csv & cross_validation_results.csv & model_comparison.csv & predictions.csv")
    print("  - results/confusion_matrix_random_forest.png & confusion_matrix_xgboost.png")
    print("  - results/roc_curve_comparison.png & pr_curve_comparison.png & metric_comparison.png")
    print("  - results/random_forest_feature_importance.png & xgboost_feature_importance.png")

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY WITH ZERO DATA LEAKAGE!")
    print("=" * 75)


if __name__ == "__main__":
    main()
