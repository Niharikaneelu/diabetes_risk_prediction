"""
Model training, hyperparameter optimization, and leak-free cross-validation module
for Diabetes Risk Prediction.

Implements:
1. Leak-Free 5-Fold Stratified Cross-Validation on training data
2. Out-of-fold probability generation and F1-maximizing threshold optimization
3. Random Forest Classifier with balanced class weighting
4. XGBoost Classifier with positive-class scale weighting
5. Multi-metric evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ml.model_utils import ZERO_AS_MISSING


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
    **kwargs: Any,
) -> RandomForestClassifier:
    """
    Train an optimized Random Forest classifier on training data.
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "class_weight": class_weight,
        "random_state": random_state,
        "n_jobs": 1,
    }
    default_params.update(kwargs)
    rf = RandomForestClassifier(**default_params)
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    scale_pos_weight: Optional[float] = None,
    **kwargs: Any,
) -> XGBClassifier:
    """
    Train an optimized XGBoost classifier on training data.
    """
    if scale_pos_weight is None:
        neg_count = float((y_train == 0).sum())
        pos_count = float((y_train == 1).sum())
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    default_params = {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": 1,
    }
    default_params.update(kwargs)
    xgb = XGBClassifier(**default_params)
    xgb.fit(X_train, y_train)
    return xgb


def cross_validate_models_leak_free(
    model_factories: Dict[str, Callable[[pd.DataFrame, pd.Series], Any]],
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """
    Perform rigorous 5-Fold Stratified Cross-Validation on the training data with strict
    fold-level imputation isolation (Zero Leakage).

    For every fold:
      1. Imputation medians are fitted SOLELY on the training fold.
      2. Both training and validation folds are imputed with fold medians.
      3. Model is trained and out-of-fold validation probabilities are generated.
      4. Fold metrics are calculated.

    Returns:
      (cv_summary_df, per_fold_metrics_dict, oof_probabilities_dict)
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    n_samples = len(X_train_raw)

    summary_records = []
    per_fold_dict = {}
    oof_probs_dict = {}

    for name, factory in model_factories.items():
        oof_probs = np.zeros(n_samples)
        fold_records = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_raw, y_train), 1):
            X_tr = X_train_raw.iloc[train_idx].copy()
            y_tr = y_train.iloc[train_idx].copy()
            X_va = X_train_raw.iloc[val_idx].copy()
            y_va = y_train.iloc[val_idx].copy()

            # Leak-free imputation: fit medians strictly on X_tr
            fold_medians = X_tr[ZERO_AS_MISSING].median()
            X_tr = X_tr.fillna(fold_medians)
            X_va = X_va.fillna(fold_medians)

            # Fit model
            model = factory(X_tr, y_tr)
            model.fit(X_tr, y_tr)

            # Predict probabilities on validation fold
            val_probs = model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = val_probs

            # Standard threshold = 0.5 fold evaluation
            val_preds = (val_probs >= 0.5).astype(int)

            acc = accuracy_score(y_va, val_preds)
            prec = precision_score(y_va, val_preds, zero_division=0)
            rec = recall_score(y_va, val_preds, zero_division=0)
            f1 = f1_score(y_va, val_preds, zero_division=0)
            auc = roc_auc_score(y_va, val_probs)
            pr_auc = average_precision_score(y_va, val_probs)

            fold_records.append({
                "Fold": fold,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "ROC_AUC": auc,
                "PR_AUC": pr_auc,
            })

        df_fold_results = pd.DataFrame(fold_records)
        per_fold_dict[name] = df_fold_results
        oof_probs_dict[name] = oof_probs

        summary_records.append({
            "Model": name,
            "Accuracy_Mean": float(df_fold_results["Accuracy"].mean()),
            "Accuracy_Std": float(df_fold_results["Accuracy"].std()),
            "Precision_Mean": float(df_fold_results["Precision"].mean()),
            "Precision_Std": float(df_fold_results["Precision"].std()),
            "Recall_Mean": float(df_fold_results["Recall"].mean()),
            "Recall_Std": float(df_fold_results["Recall"].std()),
            "F1_Score_Mean": float(df_fold_results["F1_Score"].mean()),
            "F1_Score_Std": float(df_fold_results["F1_Score"].std()),
            "ROC_AUC_Mean": float(df_fold_results["ROC_AUC"].mean()),
            "ROC_AUC_Std": float(df_fold_results["ROC_AUC"].std()),
            "PR_AUC_Mean": float(df_fold_results["PR_AUC"].mean()),
            "PR_AUC_Std": float(df_fold_results["PR_AUC"].std()),
        })

    cv_summary_df = pd.DataFrame(summary_records)
    return cv_summary_df, per_fold_dict, oof_probs_dict


def optimize_threshold_oof(
    y_true: Union[pd.Series, np.ndarray],
    oof_probs: np.ndarray,
    threshold_min: float = 0.20,
    threshold_max: float = 0.80,
    threshold_step: float = 0.02,
) -> Dict[str, Any]:
    """
    Optimize decision threshold exclusively on training-side Out-Of-Fold probabilities.
    Maximizes positive-class F1-score without touching holdout data.

    Returns dictionary with:
      - optimal_threshold
      - oof_f1_optimal
      - oof_recall_optimal
      - oof_precision_optimal
      - oof_f1_default (threshold = 0.5)
      - oof_recall_default
      - oof_precision_default
      - f1_gain
      - threshold_curve (DataFrame of evaluated thresholds and scores)
    """
    y_arr = np.array(y_true).astype(int)
    thresholds = np.round(np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step), 2)
    # Ensure 0.5 is evaluated
    if 0.50 not in thresholds:
        thresholds = np.sort(np.append(thresholds, 0.50))

    curve = []
    best_th = 0.50
    best_f1 = -1.0

    for th in thresholds:
        preds = (oof_probs >= th).astype(int)
        f1 = float(f1_score(y_arr, preds, zero_division=0))
        rec = float(recall_score(y_arr, preds, zero_division=0))
        prec = float(precision_score(y_arr, preds, zero_division=0))
        acc = float(accuracy_score(y_arr, preds))

        curve.append({
            "Threshold": float(th),
            "F1_Score": f1,
            "Recall": rec,
            "Precision": prec,
            "Accuracy": acc,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)

    df_curve = pd.DataFrame(curve)

    # Metrics at optimal threshold
    best_preds = (oof_probs >= best_th).astype(int)
    opt_f1 = float(f1_score(y_arr, best_preds, zero_division=0))
    opt_rec = float(recall_score(y_arr, best_preds, zero_division=0))
    opt_prec = float(precision_score(y_arr, best_preds, zero_division=0))

    # Metrics at default 0.50 threshold
    def_preds = (oof_probs >= 0.50).astype(int)
    def_f1 = float(f1_score(y_arr, def_preds, zero_division=0))
    def_rec = float(recall_score(y_arr, def_preds, zero_division=0))
    def_prec = float(precision_score(y_arr, def_preds, zero_division=0))

    return {
        "optimal_threshold": float(best_th),
        "oof_f1_optimal": opt_f1,
        "oof_recall_optimal": opt_rec,
        "oof_precision_optimal": opt_prec,
        "oof_f1_default": def_f1,
        "oof_recall_default": def_rec,
        "oof_precision_default": def_prec,
        "f1_gain": float(opt_f1 - def_f1),
        "threshold_curve": df_curve,
    }
