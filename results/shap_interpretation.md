# SHAP Model Explainability & Clinical Risk Interpretation Report

**Project:** Explainable Diabetes Risk Prediction Using Machine Learning  
**Explained Model:** XGBClassifier (Selected Best Model)  
**Evaluation Set:** Untouched Test Partition ($N = 154$ patients, 8 clinical features)  
**Explainer Algorithm:** SHAP `TreeExplainer` (Tree SHAP exact formulation)

---

## 1. Executive Summary & Top Global Risk Factors

Global explainability reveals that the trained XGBoost model relies predominantly on metabolic and demographic biomarkers when assessing diabetes risk. 

According to mean absolute SHAP values ($\mathbb{E}[|\phi_i|]$), the top global predictive features are:
1. **Glucose** (Mean |SHAP| = 0.9319)
2. **BMI** (Mean |SHAP| = 0.5229)
3. **Age** (Mean |SHAP| = 0.2772)
4. **DiabetesPedigreeFunction** (Mean |SHAP| = 0.2310)

Together, **Glucose**, **BMI**, and **Age** account for the overwhelming majority of prediction variance in the model.

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
| **Glucose** | 0.3234 | 0.9319 | 1 | 1 |
| **BMI** | 0.1521 | 0.5229 | 2 | 2 |
| **Age** | 0.1235 | 0.2772 | 3 | 3 |
| **DiabetesPedigreeFunction** | 0.0882 | 0.2310 | 5 | 4 |
| **Pregnancies** | 0.0927 | 0.1522 | 4 | 5 |
| **Insulin** | 0.0803 | 0.1344 | 6 | 6 |
| **SkinThickness** | 0.0721 | 0.0548 | 7 | 7 |
| **BloodPressure** | 0.0678 | 0.0416 | 8 | 8 |

### Key Insights from Comparison:
- **Consensus at the Top:** Both XGBoost native gain and SHAP identify **Glucose** (Rank 1), **BMI** (Rank 2), and **Age** (Rank 3) as the primary predictive drivers.
- **Ranking Discrepancies:** Native tree importance evaluates splits based solely on training loss reduction, often favoring features utilized near tree roots. In contrast, SHAP evaluates marginal contributions across all feature coalitions on unseen test data, providing a more balanced assessment of test-time generalization influence.

---

## 4. Local Patient Explanations (Waterfall Analysis)

Local explainability allows clinicians to examine why the model produced a specific prediction for an individual patient.

### Case 1: High-Risk Patient (Predicted Probability: 95.3%)
- **Base Log-Odds ($E[f(x)]$):** 0.0079
- **Top Risk-Increasing Factors:**
  - **Glucose** = 171.0 (SHAP contribution: +2.288)
  - **BMI** = 43.6 (SHAP contribution: +0.466)
  - **Insulin** = 125.0 (SHAP contribution: +0.161)

- **Outcome:** The cumulative positive contributions pushed the model's output significantly above the decision threshold, resulting in a high-risk prediction (95.3%).

### Case 2: Low-Risk Patient (Predicted Probability: 5.2%)
- **Base Log-Odds ($E[f(x)]$):** 0.0079
- **Top Protective Factors:**
  - **Glucose** = 91.0 (SHAP contribution: -1.152)
  - **BMI** = 25.2 (SHAP contribution: -0.826)
  - **Age** = 23.0 (SHAP contribution: -0.312)

- **Outcome:** Normal physiological measurements provided protective contributions, keeping the model output well below the risk threshold (5.2%).

---

## 5. Methodological & Clinical Limitations

> [!IMPORTANT]
> **Clinical & Algorithmic Disclaimer:**
> 1. **Model Attribution $\neq$ Biological Causation:** SHAP explains the statistical behavior and feature weighting of the trained machine learning model. It does not establish clinical causality or etiology.
> 2. **Observational Dataset:** The Pima Indians Diabetes dataset reflects specific demographic, genetic, and geographic characteristics. Predictions and explanations may not generalize directly to diverse global patient populations without prospective recalibration.
> 3. **Non-Diagnostic Tool:** This analysis is intended exclusively for educational, translational, and machine learning interpretability research. It must never be used as a standalone diagnostic system or for prescribing medical therapy without licensed clinician oversight.
