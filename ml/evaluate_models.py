"""
Model evaluation, comparison, selection, and visualization module.

Evaluates trained Random Forest and XGBoost models on untouched test data,
generates comparison tables, publication-quality plots, extracts feature importances,
and persists leak-free model artifacts and structured metadata.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.model_utils import (
    FEATURE_COLUMNS,
    generate_confusion_matrix,
    generate_feature_importance_plot,
    generate_metric_comparison,
    generate_pr_curve,
    generate_roc_curve,
    get_project_root,
    save_models,
    save_results,
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluate a fitted model on the test partition at a specified classification threshold.

    Calculates:
      - Accuracy
      - Precision
      - Recall
      - F1-Score
      - ROC-AUC
      - PR-AUC (Average Precision)
      - Confusion Matrix
      - Classification Report
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0)

    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "Predictions": y_pred,
        "Probabilities": y_prob,
        "Confusion_Matrix": cm,
        "Classification_Report": cr,
    }


def select_best_model_from_cv(
    cv_summary_df: pd.DataFrame,
    threshold_results: Dict[str, Dict[str, Any]],
) -> str:
    """
    Select the best-performing model strictly based on training-side cross-validation
    and out-of-fold metrics (Primary: F1-score, Secondary: ROC-AUC and stability).
    Eliminates test-set leakage in model selection.
    """
    scores = {}
    for _, row in cv_summary_df.iterrows():
        name = row["Model"]
        # Use training-side OOF F1 with optimized threshold if available, else CV mean F1
        th_info = threshold_results.get(name, {})
        oof_f1 = th_info.get("oof_f1_optimal", row["F1_Score_Mean"])
        roc_auc = row["ROC_AUC_Mean"]
        # Primary selection: positive-class F1 supported by ROC-AUC
        composite = 0.70 * oof_f1 + 0.30 * roc_auc
        scores[name] = composite

    best_model_name = max(scores, key=scores.get)
    return best_model_name


def compare_models(
    eval_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Create a structured holdout evaluation comparison table.
    """
    rows = []
    for name, res in eval_results.items():
        rows.append({
            "Model": name,
            "Threshold": round(res["Threshold"], 2),
            "Accuracy": round(res["Accuracy"], 4),
            "Precision": round(res["Precision"], 4),
            "Recall": round(res["Recall"], 4),
            "F1_Score": round(res["F1_Score"], 4),
            "ROC_AUC": round(res["ROC_AUC"], 4),
            "PR_AUC": round(res.get("PR_AUC", 0.0), 4),
        })
    return pd.DataFrame(rows)


def evaluate_and_save_pipeline(
    selected_models: Dict[str, Any],
    best_model_name: str,
    cv_summary_df: pd.DataFrame,
    threshold_results: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    imputation_medians: Dict[str, float],
    output_dirs: List[Path],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Retrain selected models on complete training set, perform a single-pass evaluation
    on the untouched holdout test set using frozen thresholds, generate all comparison
    plots, export structured metadata.json, and persist everything.
    """
    eval_results = {}
    eval_results_def05 = {}
    probs_dict = {}
    predictions_dict = {}

    for name, model in selected_models.items():
        # Retrain on full training set
        model.fit(X_train, y_train)

        # Get frozen optimal threshold selected from training OOF
        opt_th = threshold_results.get(name, {}).get("optimal_threshold", 0.50)

        # Evaluate on untouched holdout at frozen optimal threshold
        res_opt = evaluate_model(model, X_test, y_test, model_name=name, threshold=opt_th)
        eval_results[name] = res_opt

        # Also evaluate at default 0.50 threshold for direct baseline comparison
        res_def = evaluate_model(model, X_test, y_test, model_name=f"{name} (th=0.50)", threshold=0.50)
        eval_results_def05[name] = res_def

        probs_dict[name] = res_opt["Probabilities"]
        slug = name.replace(" ", "_")
        predictions_dict[f"{slug}_Prediction"] = res_opt["Predictions"]
        predictions_dict[f"{slug}_Probability"] = res_opt["Probabilities"]

    # Model comparison table on holdout test set
    comparison_df = compare_models(eval_results)

    # Metrics DataFrame
    metrics_df = comparison_df[["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]].copy()

    # Predictions DataFrame
    pred_data = {"Actual": y_test.values}
    pred_data.update(predictions_dict)
    predictions_df = pd.DataFrame(pred_data)

    # Save CSV tables to all output directories
    save_results(metrics_df, cv_summary_df, comparison_df, predictions_df, output_dirs)

    # Generate visual artifacts across all output directories
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Confusion matrices (at optimized thresholds)
        for name, res in eval_results.items():
            slug = name.lower().replace(" ", "_")
            generate_confusion_matrix(
                y_true=y_test,
                y_pred=res["Predictions"],
                model_name=name,
                threshold=res["Threshold"],
                output_path=out_dir / f"confusion_matrix_{slug}.png",
            )

        # 2. Combined ROC curve
        generate_roc_curve(
            y_true=y_test,
            probs_dict=probs_dict,
            output_path=out_dir / "roc_curve_comparison.png",
        )

        # 3. Combined Precision-Recall curve
        generate_pr_curve(
            y_true=y_test,
            probs_dict=probs_dict,
            output_path=out_dir / "pr_curve_comparison.png",
        )

        # 4. Metric comparison chart
        generate_metric_comparison(
            comparison_df=comparison_df,
            output_path=out_dir / "metric_comparison.png",
        )

        # 5. Feature importances
        for name, model in selected_models.items():
            slug = name.lower().replace(" ", "_")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                generate_feature_importance_plot(
                    feature_names=FEATURE_COLUMNS,
                    importances=importances,
                    model_name=name,
                    output_path=out_dir / f"{slug}_feature_importance.png",
                )

    # Build comprehensive metadata.json
    import sklearn
    import xgboost as xgb_lib

    metadata = {
        "project": "Explainable Diabetes Risk Prediction using Machine Learning",
        "best_model_name": best_model_name,
        "random_state": random_state,
        "feature_columns": FEATURE_COLUMNS,
        "imputation_strategy": "median",
        "imputation_medians": imputation_medians,
        "missingness_indicators": False,
        "models": {
            name: {
                "class": type(model).__name__,
                "params": model.get_params(),
                "optimal_threshold": threshold_results.get(name, {}).get("optimal_threshold", 0.50),
                "oof_f1_optimal": threshold_results.get(name, {}).get("oof_f1_optimal"),
                "oof_f1_default": threshold_results.get(name, {}).get("oof_f1_default"),
                "f1_gain": threshold_results.get(name, {}).get("f1_gain"),
                "holdout_metrics": {
                    "accuracy": eval_results[name]["Accuracy"],
                    "precision": eval_results[name]["Precision"],
                    "recall": eval_results[name]["Recall"],
                    "f1_score": eval_results[name]["F1_Score"],
                    "roc_auc": eval_results[name]["ROC_AUC"],
                    "pr_auc": eval_results[name]["PR_AUC"],
                },
                "holdout_metrics_default_th05": {
                    "accuracy": eval_results_def05[name]["Accuracy"],
                    "precision": eval_results_def05[name]["Precision"],
                    "recall": eval_results_def05[name]["Recall"],
                    "f1_score": eval_results_def05[name]["F1_Score"],
                    "roc_auc": eval_results_def05[name]["ROC_AUC"],
                }
            }
            for name, model in selected_models.items()
        },
        "environment": {
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb_lib.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }
    }

    # Persist trained model artifacts & metadata
    project_root = get_project_root()
    model_dirs = [project_root / "models", project_root / "ml" / "models"]
    save_models(selected_models, best_model_name, model_dirs, metadata=metadata)

    return {
        "comparison_df": comparison_df,
        "metrics_df": metrics_df,
        "cv_summary_df": cv_summary_df,
        "predictions_df": predictions_df,
        "best_model_name": best_model_name,
        "eval_results": eval_results,
        "metadata": metadata,
    }
