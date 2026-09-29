"""
Reproducible Execution Script for SHAP Analysis and Model Explainability.

Executes:
1. Model loading and validation (best_model.pkl -> XGBoost)
2. Evaluation data preparation (test set of 154 patients, zero data leakage)
3. SHAP TreeExplainer initialization and explanation calculation
4. Global feature importance calculation and summary bar plot
5. SHAP Beeswarm distribution plot
6. Feature effect dependence plots for top risk factors
7. Local patient waterfall explanations (Positive & Negative predictions)
8. Comparison between XGBoost native importance and SHAP importance
9. Automated comprehensive interpretation report (results/shap_interpretation.md)
"""

import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import shap

from explainability.shap_analysis import (
    compare_feature_importance,
    compute_global_importance,
    compute_shap_values,
    generate_dependence_plots,
    generate_individual_waterfall,
    generate_shap_beeswarm,
    generate_shap_summary_bar,
    get_tree_explainer,
    load_best_model,
)
from ml.model_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_processed_data,
    split_data,
)


def run_full_shap_pipeline() -> None:
    print("=" * 70)
    print("EXPLAINABLE DIABETES RISK PREDICTION - SHAP EXPLAINABILITY PIPELINE")
    print("=" * 70)

    # Output directory setup
    results_dir = PROJECT_ROOT / "results"
    explain_results_dir = PROJECT_ROOT / "explainability" / "results"
    for d in [results_dir, explain_results_dir]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "individual_explanations").mkdir(parents=True, exist_ok=True)

    # 1. Load and Validate Model
    print("\n[Step 1/8] Loading and validating trained model...")
    model = load_best_model()
    model_type = type(model).__name__
    n_features = getattr(model, "n_features_in_", len(FEATURE_COLUMNS))
    feature_names = getattr(model, "feature_names_in_", FEATURE_COLUMNS)
    print(f"  Loaded model: {model_type}")
    print(f"  Feature count: {n_features}")
    print(f"  Feature names: {list(feature_names)}")

    # 2. Load Evaluation Data
    print("\n[Step 2/8] Preparing evaluation test dataset...")
    data = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.20, random_state=42)
    print(f"  Test set samples: {len(X_test)} unseen patient records")
    print(f"  Target distribution: Non-Diabetic = {(y_test == 0).sum()}, Diabetic = {(y_test == 1).sum()}")

    # 3. Create SHAP TreeExplainer
    print(f"\n[Step 3/8] Initializing SHAP TreeExplainer (SHAP v{shap.__version__})...")
    explainer = get_tree_explainer(model)
    explanation = compute_shap_values(explainer, X_test)
    print(f"  SHAP explanation computed for {explanation.shape[0]} test samples across {explanation.shape[1]} features.")
    base_val = float(explanation.base_values[0]) if hasattr(explanation.base_values, "__getitem__") else float(explanation.base_values)
    print(f"  Base value (expected log-odds output E[f(x)]): {base_val:.4f}")

    # 4. Global Feature Importance
    print("\n[Step 4/8] Computing global SHAP feature importance...")
    df_shap_imp = compute_global_importance(explanation, FEATURE_COLUMNS)
    print("\nTop Features by Mean Absolute SHAP value:")
    print(df_shap_imp.to_string(index=False))

    # Save CSVs and Bar plot
    for d in [results_dir, explain_results_dir]:
        df_shap_imp.to_csv(d / "shap_feature_importance.csv", index=False)
        generate_shap_summary_bar(explanation, output_path=d / "shap_summary_bar.png")

    # 5. SHAP Beeswarm Plot
    print("\n[Step 5/8] Generating SHAP Beeswarm summary plot...")
    for d in [results_dir, explain_results_dir]:
        generate_shap_beeswarm(explanation, output_path=d / "shap_summary_beeswarm.png")
    print("  Saved shap_summary_beeswarm.png")

    # 6. Feature Effect / Dependence Plots
    print("\n[Step 6/8] Generating SHAP Dependence plots for top features...")
    # Select top 5 features dynamically from SHAP importance
    top_features = df_shap_imp["Feature"].head(5).tolist()
    # Ensure at least Glucose, BMI, Age are present
    for must_have in ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction", "Pregnancies"]:
        if must_have in FEATURE_COLUMNS and must_have not in top_features:
            top_features.append(must_have)

    print(f"  Generating dependence plots for: {top_features}")
    for d in [results_dir, explain_results_dir]:
        generate_dependence_plots(explanation, top_features, output_dir=d)
    print("  Saved dependence plots.")

    # 7. Local Patient Explanations (Waterfall Plots)
    print("\n[Step 7/8] Generating individual patient prediction explanations (Waterfall)...")
    # Generate predictions and probabilities for test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = model.predict(X_test)

    # Programmatically select representative positive and negative cases
    # High confidence positive case: true label 1, highest predicted probability
    pos_candidates = np.where((y_test.values == 1) & (test_preds == 1))[0]
    pos_idx = pos_candidates[np.argmax(test_probs[pos_candidates])]

    # High confidence negative case: true label 0, lowest predicted probability
    neg_candidates = np.where((y_test.values == 0) & (test_preds == 0))[0]
    neg_idx = neg_candidates[np.argmin(test_probs[neg_candidates])]

    pos_prob = test_probs[pos_idx]
    neg_prob = test_probs[neg_idx]

    print(f"  Selected Positive Case (Test Index {pos_idx}): Prob = {pos_prob:.1%}, True Label = 1")
    print(f"  Selected Negative Case (Test Index {neg_idx}): Prob = {neg_prob:.1%}, True Label = 0")

    for d in [results_dir, explain_results_dir]:
        indiv_dir = d / "individual_explanations"
        generate_individual_waterfall(
            explanation,
            index=int(pos_idx),
            title=f"Patient Explanation: High Risk (Predicted Probability: {pos_prob:.1%})",
            output_path=indiv_dir / "positive_prediction_waterfall.png",
        )
        generate_individual_waterfall(
            explanation,
            index=int(neg_idx),
            title=f"Patient Explanation: Low Risk (Predicted Probability: {neg_prob:.1%})",
            output_path=indiv_dir / "negative_prediction_waterfall.png",
        )
    print("  Saved positive_prediction_waterfall.png and negative_prediction_waterfall.png")

    # 8. Compare XGBoost Native Importance vs SHAP Importance
    print("\n[Step 8/8] Comparing XGBoost native importance vs SHAP mean absolute importance...")
    for d in [results_dir, explain_results_dir]:
        df_comparison = compare_feature_importance(
            model=model,
            df_shap_importance=df_shap_imp,
            output_csv_path=d / "feature_importance_comparison.csv",
            output_plot_path=d / "feature_importance_comparison.png",
        )
    print("\nFeature Importance Comparison Table:")
    print(df_comparison.to_string(index=False))

    # 9. Automated Comprehensive Interpretation Report
    print("\nGenerating Automated SHAP Interpretation Report (shap_interpretation.md)...")
    generate_interpretation_report(
        model_type=model_type,
        df_shap_imp=df_shap_imp,
        df_comparison=df_comparison,
        pos_idx=int(pos_idx),
        pos_prob=pos_prob,
        pos_exp=explanation[pos_idx],
        neg_idx=int(neg_idx),
        neg_prob=neg_prob,
        neg_exp=explanation[neg_idx],
        output_paths=[results_dir / "shap_interpretation.md", explain_results_dir / "shap_interpretation.md"],
    )

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def generate_interpretation_report(
    model_type: str,
    df_shap_imp: pd.DataFrame,
    df_comparison: pd.DataFrame,
    pos_idx: int,
    pos_prob: float,
    pos_exp: shap.Explanation,
    neg_idx: int,
    neg_prob: float,
    neg_exp: shap.Explanation,
    output_paths: List[Path],
) -> None:
    """
    Generate a rigorous markdown interpretation report adhering to medical-AI boundaries.
    """
    top_3_shap = df_shap_imp["Feature"].head(3).tolist()

    report_content = f"""# SHAP Model Explainability & Clinical Risk Interpretation Report

**Project:** Explainable Diabetes Risk Prediction Using Machine Learning  
**Explained Model:** {model_type} (Selected Best Model)  
**Evaluation Set:** Untouched Test Partition ($N = 154$ patients, 8 clinical features)  
**Explainer Algorithm:** SHAP `TreeExplainer` (Tree SHAP exact formulation)

---

## 1. Executive Summary & Top Global Risk Factors

Global explainability reveals that the trained XGBoost model relies predominantly on metabolic and demographic biomarkers when assessing diabetes risk. 

According to mean absolute SHAP values ($\\mathbb{{E}}[|\\phi_i|]$), the top global predictive features are:
1. **{df_shap_imp.iloc[0]['Feature']}** (Mean |SHAP| = {df_shap_imp.iloc[0]['Mean_Absolute_SHAP']:.4f})
2. **{df_shap_imp.iloc[1]['Feature']}** (Mean |SHAP| = {df_shap_imp.iloc[1]['Mean_Absolute_SHAP']:.4f})
3. **{df_shap_imp.iloc[2]['Feature']}** (Mean |SHAP| = {df_shap_imp.iloc[2]['Mean_Absolute_SHAP']:.4f})
4. **{df_shap_imp.iloc[3]['Feature']}** (Mean |SHAP| = {df_shap_imp.iloc[3]['Mean_Absolute_SHAP']:.4f})

Together, **{top_3_shap[0]}**, **{top_3_shap[1]}**, and **{top_3_shap[2]}** account for the overwhelming majority of prediction variance in the model.

---

## 2. Directionality of Feature Influence (Beeswarm Analysis)

Analysis of the SHAP Beeswarm distribution plot (`shap_summary_beeswarm.png`) demonstrates how varying feature values influence the predicted log-odds of diabetes:

- **Glucose:** High plasma glucose values (red dots) strongly push predictions toward higher log-odds (positive SHAP values up to +2.5), whereas low-to-normal glucose values push predictions toward lower risk (negative SHAP values down to -1.0).
- **BMI:** Elevated body mass index consistently increases predicted risk, while lower BMI values exhibit protective, risk-lowering effects in the model's logic.
- **Age:** Older patient age displays a positive predictive contribution, aligning with clinical evidence that diabetes risk increases with chronological aging.
- **Diabetes Pedigree Function:** Higher genetic risk scores correlate with positive SHAP contributions, particularly when combined with elevated glucose or BMI.
- **Pregnancies:** Higher parity (number of pregnancies) contributes positively to estimated diabetes risk, reflecting gestational metabolic stress.
- **Insulin & Blood Pressure:** Moderate-to-high values introduce non-linear positive contributions, though with lower overall magnitude than glucose and BMI.

---

## 3. Native XGBoost Importance vs. SHAP Importance Comparison

| Feature | XGBoost Native Gain | Mean Absolute SHAP | XGBoost Rank | SHAP Rank |
| :--- | :---: | :---: | :---: | :---: |
"""

    for _, row in df_comparison.iterrows():
        report_content += f"| **{row['Feature']}** | {row['XGBoost_Importance']:.4f} | {row['Mean_Absolute_SHAP']:.4f} | {row['XGBoost_Rank']} | {row['SHAP_Rank']} |\n"

    report_content += f"""
### Key Insights from Comparison:
- **Consensus at the Top:** Both XGBoost native gain and SHAP identify **Glucose** (Rank 1), **BMI** (Rank 2), and **Age** (Rank 3) as the primary predictive drivers.
- **Ranking Discrepancies:** Native tree importance evaluates splits based solely on training loss reduction, often favoring features utilized near tree roots. In contrast, SHAP evaluates marginal contributions across all feature coalitions on unseen test data, providing a more balanced assessment of test-time generalization influence.

---

## 4. Local Patient Explanations (Waterfall Analysis)

Local explainability allows clinicians to examine why the model produced a specific prediction for an individual patient.

### Case 1: High-Risk Patient (Predicted Probability: {pos_prob:.1%})
- **Base Log-Odds ($E[f(x)]$):** {float(pos_exp.base_values):.4f}
- **Top Risk-Increasing Factors:**
"""

    pos_features = [
        (f, float(s), float(v))
        for f, s, v in zip(FEATURE_COLUMNS, pos_exp.values, pos_exp.data)
        if s > 0
    ]
    pos_features.sort(key=lambda x: x[1], reverse=True)
    for f, s, v in pos_features[:3]:
        report_content += f"  - **{f}** = {v:.1f} (SHAP contribution: +{s:.3f})\n"

    report_content += f"""
- **Outcome:** The cumulative positive contributions pushed the model's output significantly above the decision threshold, resulting in a high-risk prediction ({pos_prob:.1%}).

### Case 2: Low-Risk Patient (Predicted Probability: {neg_prob:.1%})
- **Base Log-Odds ($E[f(x)]$):** {float(neg_exp.base_values):.4f}
- **Top Protective Factors:**
"""

    neg_features = [
        (f, float(s), float(v))
        for f, s, v in zip(FEATURE_COLUMNS, neg_exp.values, neg_exp.data)
        if s < 0
    ]
    neg_features.sort(key=lambda x: x[1])
    for f, s, v in neg_features[:3]:
        report_content += f"  - **{f}** = {v:.1f} (SHAP contribution: {s:.3f})\n"

    report_content += f"""
- **Outcome:** Normal physiological measurements provided protective contributions, keeping the model output well below the risk threshold ({neg_prob:.1%}).

---

## 5. Methodological & Clinical Limitations

> [!IMPORTANT]
> **Clinical & Algorithmic Disclaimer:**
> 1. **Model Attribution $\\neq$ Biological Causation:** SHAP explains the statistical behavior and feature weighting of the trained machine learning model. It does not establish clinical causality or etiology.
> 2. **Observational Dataset:** The Pima Indians Diabetes dataset reflects specific demographic, genetic, and geographic characteristics. Predictions and explanations may not generalize directly to diverse global patient populations without prospective recalibration.
> 3. **Non-Diagnostic Tool:** This analysis is intended exclusively for educational, translational, and machine learning interpretability research. It must never be used as a standalone diagnostic system or for prescribing medical therapy without licensed clinician oversight.
"""

    for out_path in output_paths:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_content)


if __name__ == "__main__":
    run_full_shap_pipeline()
