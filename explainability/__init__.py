"""SHAP explainability package for diabetes risk prediction."""

from explainability.shap_analysis import (
    DiabetesExplainerService,
    compute_global_importance,
    compute_shap_values,
    explain_prediction,
    get_tree_explainer,
    load_best_model,
)

__all__ = [
    "load_best_model",
    "get_tree_explainer",
    "compute_shap_values",
    "compute_global_importance",
    "explain_prediction",
    "DiabetesExplainerService",
]
