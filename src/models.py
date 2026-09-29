"""Model construction and persistence utilities."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from preprocessing import build_preprocessor


def get_models(random_state: int = 42) -> dict:
    """Build candidate models with preprocessing included in each pipeline."""
    return {
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.01,
                        subsample=0.8,
                        colsample_bytree=1.0,
                        eval_metric="logloss",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def save_model(model, path: str | Path) -> None:
    """Persist a fitted model to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path):
    """Load a persisted model from disk."""
    return joblib.load(path)
