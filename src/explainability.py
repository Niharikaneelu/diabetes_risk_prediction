"""SHAP explainability helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import shap


from typing import Any
import matplotlib.pyplot as plt
import shap

from explainability.shap_analysis import explain_prediction


def explain_tree_model(model: Any, X, output_path: str | Path | None = None):
    """Create a SHAP beeswarm plot for a fitted tree-based pipeline or model."""
    if hasattr(model, "named_steps"):
        transformed = model.named_steps["preprocessor"].transform(X)
        classifier = model.named_steps["classifier"]
    else:
        transformed = X
        classifier = model

    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    class_values = values[1] if isinstance(values, list) else values
    shap.summary_plot(
        class_values,
        transformed,
        feature_names=X.columns if hasattr(X, "columns") else None,
        show=False,
    )
    figure = plt.gcf()
    if output_path:
        figure.savefig(output_path, bbox_inches="tight", dpi=150)
    return figure
