# Frontend Copy Audit & Scientific Accuracy Report
## Project: Explainable Diabetes Risk Prediction AI
**Repository:** `diabetes_risk_prediction`  
**Date:** October 2026  
**Status:** Audit & Corrections Complete (All 29 Verification Tests Passing)

---

## 1. Executive Summary

A comprehensive scientific audit and refinement of all user-facing copy, explainability formulas, benchmark metrics, status badges, and clinical phrasing was conducted across the Explainable Diabetes Risk Prediction web application and Python backend.

### Key Corrections Completed:
1. **Removed Fabricated Certainty Metrics:** Replaced the artificial heuristic `(75 + Math.abs(p - 50) * 0.4)% Predictive Certainty` with genuine `Predicted Risk Probability` directly derived from model inference (`predict_proba`).
2. **Eliminated Unsupported Biological & Causal Claims:** Replaced claims attributing risk to "beta-cell secretory depletion", "microvascular or metabolic stress", or "polygenic inheritance score" with scientifically cautious, model-specific attribution descriptions.
3. **Harmonized Model Thresholds & Badges:** Resolved threshold discrepancy (static 0.50 vs active model thresholds). All UI components (circular gauge center, decision badges, modal displays, and report cards) dynamically synchronize with the active model's metadata (Random Forest: 0.48, XGBoost: 0.42).
4. **Clarified SHAP Attribution Output Spaces:** Audited the underlying mathematical spaces of TreeSHAP. Labeled Random Forest explanations as probability-space ($E[f(x)] \approx 0.498$, $\sum \phi_i = p - E[f(x)]$) and XGBoost explanations as log-odds margin space ($E[f(x)] \approx 0.020$, $\sum \phi_i = \text{margin} - E[f(x)]$, $\sigma(\text{margin}) = p$). Eliminated misleading stepper arithmetic that previously mixed log-odds with probability baselines.
5. **Accurate Global Importance & Benchmark Claims:** Replaced unvalidated claims such as "Accounts for 42% global gain" and "validated across benchmarked cohorts" with precise descriptions ("~40% of normalized mean |SHAP|" and "Global feature-importance analysis for Random Forest and XGBoost").
6. **Clinical Disclaimers & Non-Diagnostic Framing:** Added explicit research prototype disclaimers across the UI, footer, modal exports, and report generators, clarifying that the tool is an educational research prototype and does not provide clinical diagnoses.

---

## 2. Itemized Copy & Terminology Audit Table

| Category | Original Wording | Revised Wording | Scientific / Technical Rationale | Files Changed |
| :--- | :--- | :--- | :--- | :--- |
| **Predictive Certainty** | `85.4% Predictive Certainty` / `(75 + Math.abs(riskPercent - 50) * 0.4).toFixed(1) + '% Predictive Certainty'` | `77.3% Predicted Probability` / `displayProb + '% Predicted Probability'` | Probability is not statistical certainty; the original formula was an invented heuristic and did not reflect calibrated Bayesian posterior certainty. | `build_frontend.py`, `frontend/index.html` |
| **Calibration Claims** | `XGBoost Calibrated` / `XGBoost calibrated` | `Random Forest Active` / `XGBoost Active` | Formal post-hoc calibration curves (e.g., Platt scaling, isotonic regression) were not independently fitted or validated; Brier score alone does not justify the label "Calibrated". | `build_frontend.py`, `frontend/index.html`, `ml/server.py` |
| **Active Threshold** | `Threshold: 0.50` (hardcoded in circular risk gauge and modal) | `Threshold: 0.48` (Random Forest) / `Threshold: 0.42` (XGBoost) | The UI previously displayed a static default of 0.50 while models used optimal validation-tuned thresholds (0.48 and 0.42). Gauge, badges, and modal now dynamically track active model metadata. | `build_frontend.py`, `frontend/index.html` |
| **Brier Score Badge** | `Brier: 0.11` (hardcoded badge on risk card) | `Threshold: 0.48` / `Threshold: 0.42` (`id="brierScore"`) | Replaced static, unverified Brier badge with explicit active classification threshold indicator. | `build_frontend.py`, `frontend/index.html` |
| **Biological Causality (Age)** | `"Advancing chronological age correlates with beta-cell secretory depletion."` | `"Age contributed positively to the model's prediction for this input."` / `"Age contributed toward lower predicted risk for this input."` | Machine learning models capture empirical statistical associations, not histological or cellular beta-cell secretion mechanisms. | `build_frontend.py`, `frontend/index.html` |
| **Biological Causality (Blood Pressure)** | `"Elevated vascular tension suggests microvascular or metabolic stress."` | `"Blood pressure contributed to the model's estimated risk."` / `"Blood pressure contributed toward lower predicted risk."` | Diastolic blood pressure in Pima Indians dataset represents a single measurement without clinical etiology of microvascular damage. | `build_frontend.py`, `frontend/index.html` |
| **Biological Causality (Pedigree)** | `"Strong polygenic inheritance score indicates familial predisposition."` | `"The diabetes pedigree function contributed positively to the model's prediction."` / `"The diabetes pedigree function contributed toward lower predicted risk."` | Diabetes pedigree function is a mathematical heuristic proxy from relatives' history, not a genotyped polygenic risk score (PRS). | `build_frontend.py`, `frontend/index.html` |
| **Biological Causality (Protective Factors)** | `"Vascular and endocrine biomarkers provide partial protective buffering."` | `"Features with negative attributions shift the prediction toward lower estimated risk."` | Replaced speculative physiological claims with precise directional description of negative SHAP attributions. | `build_frontend.py`, `frontend/index.html` |
| **Biological Causality (Glucose)** | `"Elevated plasma concentration exceeds normal glycemic clearance capacity."` | `"Plasma glucose contributed positively to the model's estimated risk."` / `"Plasma glucose contributed toward lower predicted risk."` | Restored model-centric attribution framing rather than diagnosing pancreatic clearance failure. | `build_frontend.py`, `frontend/index.html` |
| **Biological Causality (Adiposity / BMI / Skin)** | `"Weight in excess of optimal body mass promotes peripheral insulin insensitivity."` / `"Elevated triceps skinfold reflects increased subcutaneous adiposity."` | `"Body mass index contributed positively to the model's estimated risk."` / `"Triceps skinfold measurement contributed positively to the model's estimated risk."` | Replaced physiological assumptions with objective feature contribution language. | `build_frontend.py`, `frontend/index.html` |
| **SHAP Dependence Curves** | `"Accelerating contribution after age 35, reflecting gradual beta-cell secretory depletion with plateau above age 50."` | `"Accelerating positive attribution after age 35, with plateau above age 50."` | Removed cellular pathology claims from dependence curve tooltips and commentary. | `build_frontend.py`, `frontend/index.html` |
| **Global Importance Validation** | `"Global insights validated across benchmarked cohorts illustrating feature dependencies for both XGBoost and Random Forest."` | `"Global feature-importance analysis for Random Forest and XGBoost."` | The dataset is single-cohort (Pima Indians); no independent external cohorts were evaluated. | `build_frontend.py`, `frontend/index.html` |
| **Global Feature Gain Claim** | `"Accounts for 42% global gain"` | `"~40% of normalized mean \|SHAP\|"` | Feature importance is derived from mean absolute SHAP attributions ($\mathbb{E}[\|\phi_i\|]$), not tree-split gain percentages. | `build_frontend.py`, `frontend/index.html` |
| **Diagnostic Claims (Phenotype)** | `"Patient biometric markers align with positive diabetic phenotype under high oral glucose challenge."` | `"Estimated diabetes risk (XX.X%) meets or exceeds the decision threshold (0.XX)."` | Algorithmic risk scores are not phenotypic clinical diagnoses. | `build_frontend.py`, `frontend/index.html` |
| **Diagnostic Follow-up Suggestion** | `"Diagnostic Follow-up Suggestion"` / `"Recommended confirmation via laboratory HbA1c panel and fasting lipid profile given persistent elevated glucose excursion."` | `"Model Assessment Context"` / `"The model estimates elevated risk based on patient physiological inputs. Model output is an educational/research estimate and does not constitute a clinical diagnosis."` | Replaced clinical prescribing suggestions with cautious research/educational assessment context. | `build_frontend.py`, `frontend/index.html` |
| **Report Export Disclaimer** | `"This computational report is generated for investigational clinical decision support. Final diagnostic determinations must be performed by a licensed physician..."` | `"Notice: This model is an educational/research prototype and does not replace professional medical assessment or provide a clinical diagnosis. Final clinical evaluations must be performed by a qualified healthcare professional."` | Explicitly states prototype/educational research status and non-diagnostic nature. | `build_frontend.py`, `frontend/index.html` |
| **Streamlit Page Title** | `"Diabetes Risk AI • Clinical Decision Support"` | `"Diabetes Risk AI • Explainable Risk Assessment (Research Prototype)"` | Removed clinical decision support claim from browser title. | `app.py` |
| **Benchmark Matrix** | Acc: 74.0%, Prec: 59.0%, Rec: 85.2%, F1: 0.6970 across both or outdated metrics | Random Forest: Acc 73.38%, Prec 58.67%, Rec 81.48%, F1 68.22%, AUC 0.8333.<br>XGBoost: Acc 74.03%, Prec 58.97%, Rec 85.19%, F1 69.70%, AUC 0.8404. | Aligned all displayed benchmark table metrics, confusion matrices, and ROC values strictly with saved locked metadata in `models/metadata.json`. | `build_frontend.py`, `frontend/index.html` |

---

## 3. SHAP Arithmetic and Output Space Audit

### Detailed Output Space Verification

| Property | Random Forest Classifier (`random_forest.pkl`) | XGBoost Classifier (`xgboost.pkl`) |
| :--- | :--- | :--- |
| **TreeExplainer Type** | Binary class trees with probability-leaf outputs | Tree margin ensemble with log-odds margins |
| **Raw Base Value $E[f(x)]$** | `0.4975` (Class 1 probability baseline) | `0.0205` (Margin log-odds baseline) |
| **Attribution Space ($\phi_i$)** | Probability space ($\Delta p$) | Log-odds margin space ($\Delta \text{margin}$) |
| **Additivity Equation** | $E[f(x)] + \sum_{i=1}^{M} \phi_i = p(x)$ | $E[f(x)] + \sum_{i=1}^{M} \phi_i = \text{margin}(x)$ |
| **Probability Mapping** | Identity: $p(x) = f(x)$ | Logistic sigmoid: $p(x) = \frac{1}{1 + e^{-\text{margin}(x)}}$ |
| **Observed Precision** | Machine precision $\Delta < 10^{-15}$ | Floating-point precision $\Delta < 10^{-7}$ |
| **Frontend UI Treatment** | Labeled: *Cumulative Feature Contributions (Probability Space)*. Stepper adds attributions directly to probability baseline. | Labeled: *Cumulative Feature Contributions (Log-Odds Margin Space)*. Stepper tracks margin changes and maps final margin via sigmoid to probability. |

### Elimination of Misleading Probability-Stepper Arithmetic:
- **Previous issue:** The UI stepper attempted to show additive waterfall bars leading directly to probability while displaying log-odds attributions for XGBoost without transformation context.
- **Resolution:**
  - `explainability/shap_analysis.py` and `ml/server.py` now explicitly return `output_space` (`"probability"` vs `"log_odds"`) and `margin_output`.
  - The frontend dynamically updates the stepper subtitle and formula:
    - **Random Forest:** `Each feature contributes additively in probability space from baseline E[f(x)] = 0.498 to predicted risk probability p = 0.773.`
    - **XGBoost:** `Each feature contributes additively in log-odds margin space from baseline E[f(x)] = 0.020 to margin output f(x) = 1.132, mapped via logistic sigmoid to predicted probability p = 0.756.`
  - No log-odds values are added directly to probability baselines.

---

## 4. UI Inconsistency Trace & Data Flow Synchronization

| Component | Inconsistency Identified | Root Cause | Fix Applied |
| :--- | :--- | :--- | :--- |
| **Risk Gauge Threshold** | Gauge center displayed static `Threshold: 0.50`, regardless of active model. | Hardcoded HTML string `<span ...>Threshold: 0.50</span>` with no element ID. | Assigned `id="riskGaugeThreshold"`. Updated `changeModel()` and `updateRiskOutput()` to dynamically set threshold to `0.48` for Random Forest and `0.42` for XGBoost. |
| **Model Header Badge** | Banner and hero cards displayed `XGBoost Calibrated` even when Random Forest was active. | Hardcoded strings in hero markup and `changeModel()` switcher. | Initialized default active model to Random Forest (`Selected Best • Threshold: 0.48`). Hero badge displays `Random Forest Active` or `XGBoost Active`. |
| **Settings Modal Threshold** | Model selector modal listed static threshold `0.50`. | Hardcoded text in model selection radio options. | Assigned dynamic ID `modalThresholdVal` and bound threshold updates directly to active model selection. |
| **Predicted Risk Probability** | Footer displayed fabricated certainty score. | JS calculated `(75 + Math.abs(riskPercent - 50) * 0.4).toFixed(1) + '% Predictive Certainty'`. | Changed to `displayProb + '% Predicted Probability'`, reflecting genuine model inference. |

---

## 5. Verification Test Suite Results

The full test suite was executed to guarantee inference consistency, zero data leakage, and metadata synchronization:

```bash
$ pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\bless\GIT\diabetes_risk_prediction
collected 29 items

tests/test_inference.py::test_direct_model_vs_explainer_service[random_forest-case0] PASSED [  3%]
tests/test_inference.py::test_direct_model_vs_explainer_service[random_forest-case1] PASSED [  6%]
tests/test_inference.py::test_direct_model_vs_explainer_service[random_forest-case2] PASSED [ 10%]
tests/test_inference.py::test_direct_model_vs_explainer_service[random_forest-case3] PASSED [ 13%]
tests/test_inference.py::test_direct_model_vs_explainer_service[xgboost-case0] PASSED [ 17%]
tests/test_inference.py::test_direct_model_vs_explainer_service[xgboost-case1] PASSED [ 20%]
tests/test_inference.py::test_direct_model_vs_explainer_service[xgboost-case2] PASSED [ 24%]
tests/test_inference.py::test_direct_model_vs_explainer_service[xgboost-case3] PASSED [ 27%]
tests/test_inference.py::test_http_api_vs_direct_model[random_forest-case0] PASSED [ 31%]
tests/test_inference.py::test_http_api_vs_direct_model[random_forest-case1] PASSED [ 34%]
tests/test_inference.py::test_http_api_vs_direct_model[random_forest-case2] PASSED [ 37%]
tests/test_inference.py::test_http_api_vs_direct_model[random_forest-case3] PASSED [ 41%]
tests/test_inference.py::test_http_api_vs_direct_model[xgboost-case0] PASSED [ 44%]
tests/test_inference.py::test_http_api_vs_direct_model[xgboost-case1] PASSED [ 48%]
tests/test_inference.py::test_http_api_vs_direct_model[xgboost-case2] PASSED [ 51%]
tests/test_inference.py::test_http_api_vs_direct_model[xgboost-case3] PASSED [ 55%]
tests/test_inference.py::test_model_specific_thresholds PASSED           [ 58%]
tests/test_inference.py::test_zero_imputation_integrity PASSED           [ 62%]
tests/test_inference.py::test_api_error_handling PASSED                  [ 65%]
tests/test_pipeline.py::test_01_holdout_split_preservation PASSED        [ 68%]
tests/test_pipeline.py::test_02_leak_free_imputation PASSED              [ 72%]
tests/test_pipeline.py::test_03_out_of_fold_coverage PASSED              [ 75%]
tests/test_pipeline.py::test_04_threshold_selection_isolation PASSED     [ 79%]
tests/test_pipeline.py::test_05_metadata_and_threshold_used_in_inference PASSED [ 82%]
tests/test_pipeline.py::test_06_feature_consistency PASSED               [ 86%]
tests/test_pipeline.py::test_07_both_models_available PASSED             [ 89%]
tests/test_pipeline.py::test_08_model_selection_uses_validation_metrics PASSED [ 93%]
tests/test_pipeline.py::test_09_explainer_service_workflow PASSED        [ 96%]
tests/test_pipeline.py::test_10_evaluation_metric_computation PASSED     [100%]

============================= 29 passed in 21.87s =============================
```

### Residual / Unresolved Claims Check:
- Repository-wide grep searches for `"Predictive Certainty"`, `"beta-cell"`, `"vascular tension"`, `"polygenic inheritance"`, `"benchmarked cohorts"`, `"42% global gain"`, and hardcoded `"Threshold: 0.50"` returned **0 occurrences**.
- All saved model binaries (`models/*.joblib`, `models/*.pkl`), holdout partitions, and experimental cross-validation metrics remain completely untouched and preserved.

---

## 6. Summary of Verified Corrections

1. **Design & Layout Preserved:** The responsive layout, cards, color palette, gauges, and animations remain visually identical while delivering scientifically accurate data.
2. **Mathematical Truthfulness:** Both probability-space (Random Forest) and log-odds margin space (XGBoost) SHAP attributions are clearly distinguished and mathematically exact.
3. **Clinical Humility:** The interface consistently frames outputs as estimated statistical probabilities from an educational research prototype, avoiding diagnostic overreach or causal speculation.

---

## 7. Frontend Interaction Diagnostics & Repair

### Root Cause
During the scientific terminology audit, an unescaped single quote in `build_frontend.py` (`model\'s` inside Python's `f'''...'''` string) caused Python to emit an unescaped apostrophe in `frontend/index.html` at line 1207:
```javascript
desc: '<strong>Pregnancies Parity Effect:</strong> Higher pregnancy counts (above 4) contribute positively to the model's risk output.'
```
The browser's JavaScript parser encountered `SyntaxError: Unexpected identifier 's'` when loading Script 3. Consequently, the entire script execution was halted, preventing global functions (`changeModel`, `calculateRisk`, `resetForm`, `switchTab`, `loadCohort`) from being declared on the `window` scope and blocking all `DOMContentLoaded` event handlers. Additionally, three duplicate `</div>` tags after the hero section and a duplicate `<select>` tag in the settings modal were identified and cleaned.

### Corrections Applied
1. **JavaScript Syntax Fix:** Enclosed `desc` in double quotes: `"<strong>Pregnancies Parity Effect:</strong> Higher pregnancy counts (above 4) contribute positively to the model's risk output."`, resolving the syntax error.
2. **HTML DOM Structure Fix:** Removed 3 duplicate closing `</div>` tags after the hero neural mesh section and 1 duplicate `<select>` tag in the settings modal.
3. **Automated Syntax Verification:** Executed `node --check` across all extracted `<script>` blocks (0 errors) and validated HTML tag balance (0 unclosed tags, 0 mismatched tags).

### Browser-Level Interaction Verification
Live browser interaction testing was performed on `http://localhost:8501`:
1. **Model Switcher:** Clicking "XGBoost" switches the active model to `xgboost` (Threshold: 0.42), updates badges, and updates gauge metadata. Clicking "Random Forest (Best)" reverts cleanly to `random_forest` (Threshold: 0.48). Existing input values are preserved across toggles.
2. **Presets:** Clicking Case A, Case B, and Case C immediately populates patient biometric values and recalculates model risk and SHAP decompositions.
3. **Prediction Execution:** Clicking "Estimate Diabetes Risk" executes real inference against `http://127.0.0.1:8502/api/predict`, dynamically animating the circular risk gauge, risk tier badges, and waterfall decomposition.
4. **Clear Fields:** Clicking "Clear Fields" resets all 8 input values to blank and resets the circular gauge to awaiting assessment.
5. **Console Health:** Confirmed 0 JavaScript exceptions or unhandled rejections during all interactions.

