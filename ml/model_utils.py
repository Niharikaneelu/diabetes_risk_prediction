"""
Model utilities and helper functions for diabetes risk prediction ML pipeline.

Includes data loading, stratified train-test splitting without data leakage,
evaluation visualization (confusion matrix, ROC curve, feature importance),
and results persistence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

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


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_processed_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the dataset and validate schema, columns, and target values.

    Args:
        path: Path to diabetes.csv. If None, default to data/diabetes.csv.

    Returns:
        pd.DataFrame containing the raw/processed diabetes dataset.
    """
    if path is None:
        path = get_project_root() / "data" / "diabetes.csv"
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    data = pd.read_csv(data_path)

    # Validate required columns
    required_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Verify target is binary classification (0 and 1)
    unique_targets = set(data[TARGET_COLUMN].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Invalid target values detected: {unique_targets}. Expected {0, 1}.")

    return data


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split with strict leakage prevention.

    Physiologically impossible zero measurements (Glucose, BloodPressure, SkinThickness,
    Insulin, BMI) are converted to NaN. The imputation medians are computed SOLELY
    from the training set and subsequently applied to both train and test splits.

    Args:
        data: Full input DataFrame.
        test_size: Ratio of test partition (default 0.20 for 80/20 split).
        random_state: Random seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    cleaned = data.copy()

    # Convert invalid 0 values to NA
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, pd.NA)
    for col in ZERO_AS_MISSING:
        cleaned[col] = pd.to_numeric(cleaned[col])

    X = cleaned[FEATURE_COLUMNS]
    y = cleaned[TARGET_COLUMN].astype(int)

    # Stratified train/test split before fitting any imputation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Calculate medians strictly on X_train to prevent test data leakage
    train_medians = X_train[ZERO_AS_MISSING].median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # Double check no remaining NaNs
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Unexpected missing values remained after imputation.")

    return X_train, X_test, y_train, y_test


def generate_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate and save a publication-quality confusion matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)
    percentages = cm / total * 100

    labels = np.array([
        [f"{cm[0, 0]}\n({percentages[0, 0]:.1f}%)", f"{cm[0, 1]}\n({percentages[0, 1]:.1f}%)"],
        [f"{cm[1, 0]}\n({percentages[1, 0]:.1f}%)", f"{cm[1, 1]}\n({percentages[1, 1]:.1f}%)"]
    ])

    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
        yticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, pad=12, weight="bold")
    plt.xlabel("Predicted Label", fontsize=12, labelpad=8)
    plt.ylabel("Actual Label", fontsize=12, labelpad=8)
    plt.tight_layout()

    fig = plt.gcf()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def generate_roc_curve(
    y_true: Union[pd.Series, np.ndarray],
    probs_dict: Dict[str, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate a combined ROC curve comparing all models on the same plot.
    """
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(7, 6))
    sns.set_theme(style="whitegrid")

    palette = {"Random Forest": "#1f77b4", "XGBoost": "#ff7f0e"}

    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        color = palette.get(name, None)
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.4f})",
            linewidth=2.5,
            color=color,
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)", linewidth=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, labelpad=8)
    plt.title("ROC Curves Comparison: Random Forest vs XGBoost", fontsize=14, pad=12, weight="bold")
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()

    fig = plt.gcf()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def generate_metric_comparison(
    comparison_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate a grouped bar chart comparing performance across multiple metrics.
    """
    # Reshape comparison_df from wide to long for seaborn
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    df_melted = comparison_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(9, 5.5))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#2b5c8f", "#d95f02"],
    )

    plt.title("Model Performance Comparison on Test Set", fontsize=14, pad=12, weight="bold")
    plt.ylim(0.0, 1.08)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.legend(title="Model", title_fontsize="11", fontsize=10, loc="upper right")

    # Add numeric labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9.5,
                weight="semibold",
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    fig = plt.gcf()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def generate_feature_importance_plot(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate and save a horizontal bar chart of feature importances.
    """
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")

    color = "#2b5c8f" if "Random Forest" in model_name else "#d95f02"
    bars = plt.barh(df_imp["Feature"], df_imp["Importance"], color=color, alpha=0.85)

    plt.title(f"Feature Importance ({model_name})", fontsize=14, pad=12, weight="bold")
    plt.xlabel("Relative Importance Score", fontsize=12, labelpad=8)
    plt.ylabel("Feature", fontsize=12, labelpad=8)

    max_val = df_imp["Importance"].max()
    plt.xlim(0, max_val * 1.15)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + (max_val * 0.015),
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=9.5,
            weight="semibold",
        )

    plt.tight_layout()
    fig = plt.gcf()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def save_models(
    models_dict: Dict[str, object],
    best_model_name: str,
    directories: List[Union[str, Path]],
) -> None:
    """
    Persist trained models in .pkl and .joblib formats across specified directories.
    """
    for d in directories:
        target_dir = Path(d)
        target_dir.mkdir(parents=True, exist_ok=True)

        for name, model in models_dict.items():
            slug = name.lower().replace(" ", "_")
            pkl_path = target_dir / f"{slug}.pkl"
            joblib_path = target_dir / f"{slug}.joblib"

            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)
            joblib.dump(model, joblib_path)

        # Save the identified best model
        best_model = models_dict[best_model_name]
        with open(target_dir / "best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        joblib.dump(best_model, target_dir / "best_model.joblib")


def save_results(
    metrics_df: pd.DataFrame,
    cv_results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    directories: List[Union[str, Path]],
) -> None:
    """
    Save all result CSV tables across specified directories.
    """
    for d in directories:
        target_dir = Path(d)
        target_dir.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(target_dir / "metrics.csv", index=False)
        cv_results_df.to_csv(target_dir / "cross_validation_results.csv", index=False)
        comparison_df.to_csv(target_dir / "model_comparison.csv", index=False)
        predictions_df.to_csv(target_dir / "predictions.csv", index=False)
