"""
Model evaluation, comparison, selection, and visualization module.

Evaluates trained Random Forest and XGBoost models on untouched test data,
generates comparison tables, publication-quality plots, extracts feature importances,
and selects the best model.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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
) -> Dict[str, Any]:
    """
    Evaluate a fitted model on the test partition.

    Calculates:
      - Accuracy
      - Precision
      - Recall
      - F1-Score
      - ROC-AUC
      - Confusion Matrix
      - Classification Report
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "Predictions": y_pred,
        "Probabilities": y_prob,
        "Confusion_Matrix": cm,
        "Classification_Report": cr,
    }


def compare_models(
    eval_results: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, str]:
    """
    Create a structured model comparison table and determine the best-performing model.

    The model selection logic does NOT rely on Accuracy alone. It balances clinical risk
    detection by prioritizing F1-Score and ROC-AUC (composite score = 0.5 * F1 + 0.5 * ROC_AUC).

    Returns:
        (comparison_df, best_model_name)
    """
    rows = []
    scores = {}

    for name, res in eval_results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(res["Accuracy"], 4),
            "Precision": round(res["Precision"], 4),
            "Recall": round(res["Recall"], 4),
            "F1_Score": round(res["F1_Score"], 4),
            "ROC_AUC": round(res["ROC_AUC"], 4),
        })
        # Composite score weighting balanced detection and discriminative capacity
        composite = 0.5 * res["F1_Score"] + 0.5 * res["ROC_AUC"]
        scores[name] = composite

    comparison_df = pd.DataFrame(rows)

    # Determine best model dynamically from measured results
    best_model_name = max(scores, key=scores.get)

    return comparison_df, best_model_name


def evaluate_and_save_pipeline(
    selected_models: Dict[str, Any],
    cv_results_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dirs: List[Path],
) -> Dict[str, Any]:
    """
    Retrain selected models on complete training set, evaluate on untouched test set,
    generate all comparison plots and tables, and persist everything.
    """
    eval_results = {}
    probs_dict = {}
    predictions_dict = {}

    # Retrain on full training data and evaluate once on test data
    for name, model in selected_models.items():
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test, model_name=name)
        eval_results[name] = res
        probs_dict[name] = res["Probabilities"]
        predictions_dict[f"{name.replace(' ', '_')}_Prediction"] = res["Predictions"]
        predictions_dict[f"{name.replace(' ', '_')}_Probability"] = res["Probabilities"]

    # Model comparison table and dynamic best model selection
    comparison_df, best_model_name = compare_models(eval_results)

    # Metrics DataFrame matching requested specification
    metrics_df = comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]].copy()

    # Predictions DataFrame
    pred_data = {"Actual": y_test.values}
    pred_data.update(predictions_dict)
    predictions_df = pd.DataFrame(pred_data)

    # Save CSV tables to all output directories
    save_results(metrics_df, cv_results_df, comparison_df, predictions_df, output_dirs)

    # Generate visual artifacts across all output directories
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Confusion matrices
        for name, res in eval_results.items():
            slug = name.lower().replace(" ", "_")
            generate_confusion_matrix(
                y_true=y_test,
                y_pred=res["Predictions"],
                model_name=name,
                output_path=out_dir / f"confusion_matrix_{slug}.png",
            )

        # 2. Combined ROC curve
        generate_roc_curve(
            y_true=y_test,
            probs_dict=probs_dict,
            output_path=out_dir / "roc_curve_comparison.png",
        )

        # 3. Metric comparison chart
        generate_metric_comparison(
            comparison_df=comparison_df,
            output_path=out_dir / "metric_comparison.png",
        )

        # 4. Feature importances
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

    # Persist trained model artifacts
    project_root = get_project_root()
    model_dirs = [project_root / "models", project_root / "ml" / "models"]
    save_models(selected_models, best_model_name, model_dirs)

    return {
        "comparison_df": comparison_df,
        "metrics_df": metrics_df,
        "cv_results_df": cv_results_df,
        "predictions_df": predictions_df,
        "best_model_name": best_model_name,
        "eval_results": eval_results,
    }
