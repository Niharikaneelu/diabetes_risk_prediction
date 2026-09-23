"""SHAP explainability helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import shap


def explain_tree_model(model, X, output_path: str | Path | None = None):
    """Create a SHAP beeswarm plot for a fitted tree-based pipeline."""
    transformed = model.named_steps["preprocessor"].transform(X)
    classifier = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(transformed)
    class_values = values[1] if isinstance(values, list) else values
    shap.summary_plot(
        class_values,
        transformed,
        feature_names=X.columns,
        show=False,
    )
    figure = plt.gcf()
    if output_path:
        figure.savefig(output_path, bbox_inches="tight", dpi=150)
    return figure
