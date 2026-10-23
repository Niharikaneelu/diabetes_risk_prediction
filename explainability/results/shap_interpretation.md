# SHAP Model Explainability & Clinical Risk Interpretation Report

**Project:** Explainable Diabetes Risk Prediction Using Machine Learning  
**Explained Model:** RandomForestClassifier (Selected Best Model via Leak-Free CV)  
**Evaluation Set:** Untouched Test Partition ($N = 154$ patients, 8 clinical features)  
**Explainer Algorithm:** SHAP `TreeExplainer` (Tree SHAP formulation)

---

## 1. Executive Summary & Top Global Risk Factors

Global explainability reveals that the trained RandomForestClassifier relies predominantly on metabolic and demographic biomarkers when assessing diabetes risk. 

According to mean absolute SHAP values ($\mathbb{E}[|\phi_i|]$), the top global predictive features are:
1. **Glucose** (Mean |SHAP| = 0.1415)
2. **BMI** (Mean |SHAP| = 0.0869)
3. **Age** (Mean |SHAP| = 0.0573)
4. **Insulin** (Mean |SHAP| = 0.0351)

Together, **Glucose**, **BMI**, and **Age** account for the overwhelming majority of prediction variance in the model.

---

## 2. Directionality of Feature Influence (Beeswarm Analysis)

Analysis of the SHAP Beeswarm distribution plot (`shap_summary_beeswarm.png`) demonstrates how varying feature values influence the predicted log-odds of diabetes:

- **Glucose:** High plasma glucose values (red dots) strongly push predictions toward higher risk (positive SHAP values), whereas normal or lower glucose values push predictions toward lower risk (negative SHAP values).
- **BMI:** Elevated body mass index consistently increases predicted risk, while lower BMI values exhibit protective, risk-lowering effects in the model's logic.
- **Age:** Older patient age displays a positive predictive contribution, aligning with clinical evidence that diabetes risk increases with chronological aging.
- **Diabetes Pedigree Function:** Higher genetic risk scores correlate with positive SHAP contributions, particularly when combined with elevated glucose or BMI.
- **Pregnancies:** Higher parity contributes positively to estimated diabetes risk, reflecting gestational metabolic stress.
- **Insulin & Blood Pressure:** Moderate-to-high values introduce positive contributions, though with lower overall magnitude than glucose and BMI.

---

## 3. Native Model Importance vs. SHAP Importance Comparison

| Feature | RandomForestClassifier Native Importance | Mean Absolute SHAP | Native Rank | SHAP Rank |
| :--- | :---: | :---: | :---: | :---: |
| **Glucose** | 0.3408 | 0.1415 | 1 | 1 |
| **BMI** | 0.1848 | 0.0869 | 2 | 2 |
| **Age** | 0.1295 | 0.0573 | 3 | 3 |
| **Insulin** | 0.0937 | 0.0351 | 4 | 4 |
| **DiabetesPedigreeFunction** | 0.0795 | 0.0291 | 5 | 5 |
| **Pregnancies** | 0.0663 | 0.0281 | 6 | 6 |
| **SkinThickness** | 0.0558 | 0.0202 | 7 | 7 |
| **BloodPressure** | 0.0496 | 0.0105 | 8 | 8 |

### Key Insights from Comparison:
- **Consensus at the Top:** Both native tree importance and SHAP identify **Glucose** (Rank 1), **BMI** (Rank 2), and **Age** (Rank 3) as the primary predictive drivers.
- **Ranking Discrepancies:** Native tree importance evaluates splits based solely on training loss reduction, often favoring features utilized near tree roots. In contrast, SHAP evaluates marginal contributions across all feature coalitions on unseen test data, providing a more balanced assessment of test-time generalization influence.

---

## 4. Local Patient Explanations (Waterfall Analysis)

Local explainability allows clinicians to examine why the model produced a specific prediction for an individual patient.

### Case 1: High-Risk Patient (Predicted Probability: 88.2%)
- **Base Expected Value ($E[f(x)]$):** 0.4985
- **Top Risk-Increasing Factors:**
  - **Glucose** = 194.0 (SHAP contribution: +0.214)
  - **Age** = 41.0 (SHAP contribution: +0.052)
  - **BMI** = 35.9 (SHAP contribution: +0.044)

- **Outcome:** Cumulative positive contributions pushed the model's probability above the decision threshold, resulting in a high-risk prediction (88.2%).

### Case 2: Low-Risk Patient (Predicted Probability: 2.5%)
- **Base Expected Value ($E[f(x)]$):** 0.4985
- **Top Protective Factors:**
  - **Glucose** = 91.0 (SHAP contribution: -0.161)
  - **BMI** = 25.2 (SHAP contribution: -0.128)
  - **Age** = 23.0 (SHAP contribution: -0.063)

- **Outcome:** Normal physiological measurements provided protective contributions, keeping the model output well below the risk threshold (2.5%).

---

## 5. Methodological & Clinical Limitations

> [!IMPORTANT]
> **Clinical & Algorithmic Disclaimer:**
> 1. **Model Attribution $\neq$ Biological Causation:** SHAP explains the statistical behavior and feature weighting of the trained machine learning model. It does not establish clinical causality or etiology.
> 2. **Observational Dataset:** The Pima Indians Diabetes dataset reflects specific demographic, genetic, and geographic characteristics. Predictions and explanations may not generalize directly to diverse global patient populations without prospective recalibration.
> 3. **Non-Diagnostic Tool:** This analysis is intended exclusively for educational, translational, and machine learning interpretability research. It must never be used as a standalone diagnostic system or for prescribing medical therapy without licensed clinician oversight.
