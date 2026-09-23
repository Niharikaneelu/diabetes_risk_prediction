"""Data loading and preprocessing helpers for diabetes risk prediction."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the CSV and validate the expected Pima-style schema."""
    data = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Replace impossible zero measurements and return X and y."""
    cleaned = data.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, pd.NA)
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].apply(pd.to_numeric)
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].fillna(cleaned[ZERO_AS_MISSING].median())
    return cleaned[FEATURE_COLUMNS], cleaned[TARGET_COLUMN].astype(int)


def split_data(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Create a stratified train/test split."""
    features, target = prepare_features(data)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_preprocessor() -> Pipeline:
    """Return the feature preprocessing pipeline used by the models."""
    return Pipeline([("scaler", StandardScaler())])
