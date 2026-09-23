"""Model construction and persistence utilities."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


def get_models(random_state: int = 42) -> dict:
    """Build baseline models with preprocessing included in each pipeline."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300, random_state=random_state, class_weight="balanced"
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
