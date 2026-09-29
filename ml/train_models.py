"""
Model training and hyperparameter optimization module for Diabetes Risk Prediction.

Implements:
1. Baseline and Tuned Random Forest Classifier
2. Baseline and Tuned XGBoost Classifier
3. 5-Fold Stratified Cross-Validation on training data
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    **kwargs: Any,
) -> RandomForestClassifier:
    """
    Train a baseline or custom Random Forest classifier on training data.
    """
    default_params = {
        "n_estimators": 100,
        "random_state": random_state,
        "n_jobs": 1,
    }
    default_params.update(kwargs)
    rf = RandomForestClassifier(**default_params)
    rf.fit(X_train, y_train)
    return rf


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    """
    Perform controlled hyperparameter tuning for Random Forest using 5-Fold Stratified CV.
    Optimizes primarily on F1-score on the training set.

    Parameter search grid:
      - n_estimators: [100, 200, 300]
      - max_depth: [None, 5, 10, 15]
      - min_samples_split: [2, 5, 10]
      - min_samples_leaf: [1, 2, 4]
      - max_features: ['sqrt', 'log2']
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    **kwargs: Any,
) -> XGBClassifier:
    """
    Train a baseline or custom XGBoost classifier on training data.
    """
    default_params = {
        "n_estimators": 100,
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": 1,
    }
    default_params.update(kwargs)
    xgb = XGBClassifier(**default_params)
    xgb.fit(X_train, y_train)
    return xgb


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[XGBClassifier, Dict[str, Any], float]:
    """
    Perform controlled hyperparameter tuning for XGBoost using 5-Fold Stratified CV.
    Optimizes primarily on F1-score on the training set.

    Parameter search grid:
      - n_estimators: [100, 200, 300]
      - max_depth: [3, 5, 7]
      - learning_rate: [0.01, 0.05, 0.1]
      - subsample: [0.8, 1.0]
      - colsample_bytree: [0.8, 1.0]
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def cross_validate_models(
    models_dict: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Perform rigorous 5-Fold Stratified Cross-Validation on the training data.
    Computes mean and standard deviation for:
      - Accuracy
      - Precision
      - Recall
      - F1-Score
      - ROC-AUC
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    records = []

    for name, model_instance in models_dict.items():
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        roc_aucs = []

        # Clone parameters without fitting
        params = model_instance.get_params()

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_model = type(model_instance)(**params)
            fold_model.fit(X_fold_train, y_fold_train)

            y_pred = fold_model.predict(X_fold_val)
            y_prob = fold_model.predict_proba(X_fold_val)[:, 1]

            accuracies.append(accuracy_score(y_fold_val, y_pred))
            precisions.append(precision_score(y_fold_val, y_pred, zero_division=0))
            recalls.append(recall_score(y_fold_val, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_fold_val, y_pred, zero_division=0))
            roc_aucs.append(roc_auc_score(y_fold_val, y_prob))

        records.append({
            "Model": name,
            "Accuracy_Mean": float(np.mean(accuracies)),
            "Accuracy_Std": float(np.std(accuracies)),
            "Precision_Mean": float(np.mean(precisions)),
            "Precision_Std": float(np.std(precisions)),
            "Recall_Mean": float(np.mean(recalls)),
            "Recall_Std": float(np.std(recalls)),
            "F1_Score_Mean": float(np.mean(f1_scores)),
            "F1_Score_Std": float(np.std(f1_scores)),
            "ROC_AUC_Mean": float(np.mean(roc_aucs)),
            "ROC_AUC_Std": float(np.std(roc_aucs)),
        })

    return pd.DataFrame(records)
