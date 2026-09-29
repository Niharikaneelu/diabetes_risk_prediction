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
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, leak_free: bool = True
):
    """Create a stratified train/test split, strictly fitting imputation on train set by default."""
    if not leak_free:
        features, target = prepare_features(data)
        return train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )

    # Strictly leak-free: split raw data first, then compute medians on train set only
    cleaned = data.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, pd.NA)
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].apply(pd.to_numeric)

    X = cleaned[FEATURE_COLUMNS]
    y = cleaned[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit imputation medians strictly on X_train, apply to both X_train and X_test
    train_medians = X_train[ZERO_AS_MISSING].median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> Pipeline:
    """Return the feature preprocessing pipeline used by the models."""
    return Pipeline([("scaler", StandardScaler())])
