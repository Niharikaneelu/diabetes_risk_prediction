"""
SHAP Analysis and Model Explainability Module for Diabetes Risk Prediction.

Implements:
1. SHAP TreeExplainer initialization and multi-model support (Random Forest & XGBoost)
2. Global feature importance calculation (Mean Absolute SHAP)
3. SHAP Beeswarm summary plots and bar plots
4. SHAP Dependence / Feature Effect plots
5. Local individual prediction explanations (Waterfall plots)
6. Comparison of native model feature importance vs SHAP importance
7. Reusable explain_prediction API with metadata-driven imputation and thresholding
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from ml.model_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ZERO_AS_MISSING,
    get_project_root,
    load_metadata,
    load_processed_data,
    split_data,
)


def load_best_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """
    Load the best trained model (models/best_model.pkl).
    Accepts Random Forest or XGBoost.
    """
    if model_path is None:
        model_path = get_project_root() / "models" / "best_model.pkl"
    path = Path(model_path)
    if not path.exists():
        path = path.with_suffix(".joblib")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model


def load_any_model(model_name: str = "best") -> Any:
    """
    Load a named model from the models/ directory.
    Supported names: 'best', 'xgboost', 'random_forest'.
    """
    name_map = {
        "best": "best_model",
        "xgboost": "xgboost",
        "random_forest": "random_forest",
    }
    key = model_name.lower().strip()
    filename = name_map.get(key, "best_model")
    model_path = get_project_root() / "models" / f"{filename}.pkl"
    if not model_path.exists():
        model_path = model_path.with_suffix(".joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """
    Initialize a SHAP TreeExplainer for tree-based classifiers (Random Forest or XGBoost).
    """
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> shap.Explanation:
    """
    Compute SHAP Explanation object for the given dataset.
    Ensures column ordering strictly matches training schema.
    For binary multi-output models (e.g. Random Forest), isolates the positive class (Class 1 = Diabetic).
    """
    X_ordered = X[FEATURE_COLUMNS]
    explanation = explainer(X_ordered)
    if hasattr(explanation, "values") and explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    return explanation


def compute_global_importance(
    explanation: shap.Explanation,
    feature_names: List[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP value for each feature, sorted descending.
    """
    vals = explanation.values
    if isinstance(vals, list):
        vals = vals[1]
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap,
    })
    df_importance = df_importance.sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    return df_importance


def generate_shap_summary_bar(
    explanation: shap.Explanation,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate and save a clean bar plot of global mean absolute SHAP values.
    """
    plt.figure(figsize=(8, 5))
    shap.plots.bar(explanation, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=11)
    plt.tight_layout()

    fig = plt.gcf()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def generate_shap_beeswarm(
    explanation: shap.Explanation,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate and save the SHAP beeswarm summary plot illustrating
    feature magnitude, impact direction, and distribution.
    """
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(explanation, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title("SHAP Beeswarm Plot: Feature Impact on Diabetes Risk", fontsize=13, weight="bold", pad=12)
    plt.xlabel("SHAP Value (Impact on Model Output / Log-Odds of Diabetes)", fontsize=11)
    plt.tight_layout()

    fig = plt.gcf()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def generate_dependence_plots(
    explanation: shap.Explanation,
    features_to_plot: List[str],
    output_dir: Union[str, Path],
) -> List[Path]:
    """
    Generate and save SHAP dependence (scatter) plots for key features.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for feat in features_to_plot:
        if feat not in FEATURE_COLUMNS:
            continue
        plt.figure(figsize=(7, 5))
        shap.plots.scatter(explanation[:, feat], color=explanation, show=False)
        plt.title(f"SHAP Dependence: {feat}", fontsize=13, weight="bold", pad=10)
        plt.ylabel("SHAP Value (Impact on Risk)", fontsize=11)
        plt.xlabel(feat, fontsize=11)
        plt.tight_layout()

        fig = plt.gcf()
        file_path = out_dir / f"shap_dependence_{feat.lower()}.png"
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(file_path)

    return saved_paths


def generate_individual_waterfall(
    explanation: shap.Explanation,
    index: int,
    title: str,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate and save a SHAP waterfall plot for a single patient record.
    """
    plt.figure(figsize=(9, 6))
    single_exp = explanation[index]
    shap.plots.waterfall(single_exp, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.tight_layout()

    fig = plt.gcf()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def compare_feature_importance(
    model: Any,
    df_shap_importance: pd.DataFrame,
    output_csv_path: Optional[Union[str, Path]] = None,
    output_plot_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Compare native model feature importances with SHAP mean absolute importances.
    Creates comparison table and publication-quality dual-ranked comparison plot.
    """
    native_importances = model.feature_importances_
    model_name = "Random Forest" if "RandomForest" in type(model).__name__ else "XGBoost"

    df_native = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        f"{model_name.replace(' ', '_')}_Importance": native_importances,
    })

    merged = pd.merge(df_native, df_shap_importance, on="Feature")

    native_col = f"{model_name.replace(' ', '_')}_Importance"
    merged["Native_Rank"] = merged[native_col].rank(ascending=False, method="min").astype(int)
    merged["SHAP_Rank"] = merged["Mean_Absolute_SHAP"].rank(ascending=False, method="min").astype(int)

    # Sort by SHAP rank
    merged = merged.sort_values(by="SHAP_Rank").reset_index(drop=True)

    if output_csv_path:
        out_csv = Path(output_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)

    if output_plot_path:
        plot_df = merged.copy()
        plot_df["Norm_Native"] = plot_df[native_col] / plot_df[native_col].max()
        plot_df["Norm_SHAP"] = plot_df["Mean_Absolute_SHAP"] / plot_df["Mean_Absolute_SHAP"].max()

        df_melt = plot_df.melt(
            id_vars=["Feature"],
            value_vars=["Norm_Native", "Norm_SHAP"],
            var_name="Importance_Type",
            value_name="Normalized_Score",
        )
        df_melt["Importance_Type"] = df_melt["Importance_Type"].replace({
            "Norm_Native": f"{model_name} Native Importance",
            "Norm_SHAP": "SHAP Mean |Value|",
        })

        palette = ["#2b5c8f", "#1f77b4"] if "Random Forest" in model_name else ["#ff7f0e", "#1f77b4"]
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(
            data=df_melt,
            x="Feature",
            y="Normalized_Score",
            hue="Importance_Type",
            palette=palette,
        )
        plt.title(f"Comparison: {model_name} Native Importance vs. SHAP Importance", fontsize=13, weight="bold", pad=12)
        plt.ylabel("Relative Score (Normalized to Max = 1.0)", fontsize=11)
        plt.xlabel("Predictor Feature", fontsize=11)
        plt.xticks(rotation=25, ha="right", fontsize=10)
        plt.legend(title="Importance Method", fontsize=10)
        plt.tight_layout()

        fig = plt.gcf()
        out_plot = Path(output_plot_path)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return merged


class DiabetesExplainerService:
    """
    Singleton service for computing real-time SHAP explanations for new inference queries.
    Uses saved metadata.json for leak-free medians and classification threshold.
    """
    _instances: Dict[str, "DiabetesExplainerService"] = {}

    def __init__(self, model_path: Optional[Union[str, Path]] = None, model_name: str = "best"):
        if model_path is not None:
            self.model = load_best_model(model_path)
        else:
            self.model = load_any_model(model_name)
        self.model_name = model_name
        self.explainer = get_tree_explainer(self.model)

        # Load metadata if present
        meta = load_metadata()
        if meta and "imputation_medians" in meta:
            self.imputation_medians = meta["imputation_medians"]
        else:
            data = load_processed_data()
            X_train, _, _, _ = split_data(data, test_size=0.20, random_state=42)
            self.imputation_medians = X_train[ZERO_AS_MISSING].median().to_dict()

        # Decision threshold
        self.threshold = 0.50
        if meta and "models" in meta:
            norm_models = {k.lower().replace("_", " "): v for k, v in meta["models"].items()}
            lookup_key = model_name.lower().replace("_", " ").strip()
            if lookup_key == "best":
                best_name = meta.get("best_model_name", "Random Forest")
                lookup_key = best_name.lower().replace("_", " ").strip()
            m_info = norm_models.get(lookup_key, {})
            self.threshold = float(m_info.get("optimal_threshold", 0.50))

    @classmethod
    def get_instance(cls, model_name: str = "best") -> "DiabetesExplainerService":
        key = model_name.lower().strip()
        if key not in cls._instances:
            cls._instances[key] = DiabetesExplainerService(model_name=key)
        return cls._instances[key]

    def explain_prediction(
        self,
        input_data: Union[Dict[str, float], pd.DataFrame, pd.Series],
    ) -> Dict[str, Any]:
        """
        Reusable inference explanation API:
        1. Validates input features
        2. Applies training median imputation
        3. Generates probability and threshold-governed predicted class
        4. Calculates exact SHAP contributions and base value
        5. Identifies top positive and negative contributors
        """
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.Series):
            df_input = pd.DataFrame([input_data.to_dict()])
        else:
            df_input = input_data.copy()

        # Check required columns
        missing = set(FEATURE_COLUMNS).difference(df_input.columns)
        if missing:
            raise ValueError(f"Input is missing required features: {sorted(missing)}")

        # Ensure correct column ordering
        df_input = df_input[FEATURE_COLUMNS].copy()

        # Convert zero to median for ZERO_AS_MISSING if zero is supplied
        for col in ZERO_AS_MISSING:
            val = df_input.at[0, col]
            if pd.isna(val) or val == 0:
                df_input.at[0, col] = self.imputation_medians[col]

        # Generate probability
        probability = float(self.model.predict_proba(df_input)[0, 1])
        predicted_class = 1 if probability >= self.threshold else 0

        # Compute SHAP
        explanation = self.explainer(df_input)

        raw_shap = explanation.values[0]
        raw_base = explanation.base_values[0]

        if np.ndim(raw_shap) == 2:
            shap_values = raw_shap[:, 1]
            base_value = float(raw_base[1])
            output_space = "probability"
        else:
            shap_values = raw_shap
            base_value = float(raw_base)
            output_space = "log_odds"

        margin_output = float(base_value + np.sum(shap_values))

        waterfall_exp = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=df_input[FEATURE_COLUMNS].values[0],
            feature_names=FEATURE_COLUMNS,
        )

        contributions = {}
        top_pos = []
        top_neg = []

        for feat, shap_val in zip(FEATURE_COLUMNS, shap_values):
            val = float(df_input[feat].iloc[0])
            sv = float(shap_val)
            contributions[feat] = {
                "feature_value": val,
                "shap_value": sv,
                "direction": "Increases Risk" if sv > 0 else "Decreases Risk",
            }
            if sv > 0:
                top_pos.append((feat, sv, val))
            else:
                top_neg.append((feat, sv, val))

        top_pos.sort(key=lambda x: x[1], reverse=True)
        top_neg.sort(key=lambda x: x[1])

        return {
            "predicted_class": predicted_class,
            "prediction_label": "Elevated Risk" if predicted_class == 1 else "Lower Risk",
            "probability": probability,
            "threshold": self.threshold,
            "base_value": base_value,
            "output_space": output_space,
            "margin_output": margin_output,
            "feature_contributions": contributions,
            "top_positive_contributors": top_pos,
            "top_negative_contributors": top_neg,
            "explanation": waterfall_exp,
            "input_df": df_input,
        }


def explain_prediction(
    input_data: Union[Dict[str, float], pd.DataFrame, pd.Series],
) -> Dict[str, Any]:
    """
    Public API function to explain an individual patient prediction.
    """
    service = DiabetesExplainerService.get_instance()
    return service.explain_prediction(input_data)
