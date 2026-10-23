# Final Verification Report: Diabetes Risk Prediction Project

**Project:** Explainable Diabetes Risk Prediction using Machine Learning  
**Target Architectures:** Random Forest & XGBoost Classifiers (strictly preserved)  
**Dataset:** Pima Indians Diabetes Dataset ($N = 769$, 8 clinical predictors)  
**Verification Date:** October 2026  
**Auditor:** Antigravity AI Code Assistant  

---

## Executive Summary & Scorecard

| Check # | Verification Task | Status | Summary Finding |
| :---: | :--- | :---: | :--- |
| **1** | **Model Selection** | **PASS** | Winner selected exclusively using training-side 5-fold cross-validation composite score; holdout metrics have zero influence. |
| **2** | **Threshold Methodology** | **PASS** | Thresholds optimized strictly on training-side Out-Of-Fold probabilities; selection metric vs. independent holdout estimate clearly distinguished. |
| **3** | **Holdout Isolation** | **PASS** | Exactly 154 records locked; imputer fitting, resampling, CV, and threshold tuning never saw holdout data. Holdout treated as consumed. |
| **4** | **Actual Application Inference** | **FAIL / PARTIAL** | **FAIL** for static embedded HTML/JS UI (`build_frontend.py` uses client-side JavaScript linear calculation); **PASS** for Python backend (`DiabetesExplainerService`). |
| **5** | **SHAP Consistency** | **PASS** | SHAP uses the selected best model, identical 8-feature schema, training-only imputation medians, and correctly handles positive-class slices. |
| **6** | **Tests and Artifacts** | **PASS** | 10/10 automated `pytest` tests pass without re-running training or accessing the holdout; all model files and metadata verified. |

---

## 1. Verify Model Selection

**Status:** **PASS**

### Audit & Evidence
- In [`ml/run_pipeline.py`](ml/run_pipeline.py#L98-L103), model selection occurs in **Step 5**, strictly before any holdout test set evaluation in **Step 6**:
  ```python
  # Step 5: Leakage-Free Model Selection (Training Metrics Only)
  best_model_name = select_best_model_from_cv(cv_summary_df, threshold_results)
  ```
- In [`ml/evaluate_models.py`](ml/evaluate_models.py#L65-L81):
  ```python
  def select_best_model_from_cv(
      cv_summary_df: pd.DataFrame,
      threshold_results: Dict[str, Dict[str, Any]],
  ) -> str:
      scores = {}
      for _, row in cv_summary_df.iterrows():
          name = row["Model"]
          th_info = threshold_results.get(name, {})
          oof_f1 = th_info.get("oof_f1_optimal", row["F1_Score_Mean"])
          roc_auc = row["ROC_AUC_Mean"]
          composite = 0.70 * oof_f1 + 0.30 * roc_auc
          scores[name] = composite
      return max(scores, key=scores.get)
  ```
- **Holdout Independence:** The holdout test set (`X_test`, `y_test`) is not passed as an argument to `select_best_model_from_cv`. The winning architecture is determined solely from training-side Out-Of-Fold F1 and CV ROC-AUC:
  - Random Forest: OOF F1 = 0.7025, CV ROC-AUC = 0.8354 $\to$ Composite = **0.7424** (Selected Winner)
  - XGBoost: OOF F1 = 0.7004, CV ROC-AUC = 0.8345 $\to$ Composite = **0.7406**
- Verified via automated test: `test_08_model_selection_uses_validation_metrics` in `tests/test_pipeline.py` passes.

---

## 2. Verify Threshold Methodology

**Status:** **PASS**

### Audit & Evidence
- **Implementation:** In [`ml/train_models.py`](ml/train_models.py#L160-L225), `optimize_threshold_oof` searches a grid of candidate cutoffs:
  - Grid: $[0.20, 0.22, 0.24, \dots, 0.80]$
  - Ground truth: `y_train` (the 615 training partition labels)
  - Predictions: `oof_probs` generated from out-of-fold validation splits across the 5 training folds
- **Selected Thresholds:**
  - **Random Forest:** Selected threshold = **$0.48$** (OOF F1 at 0.50 = 0.7017 $\to$ OOF F1 at 0.48 = **0.7025**)
  - **XGBoost:** Selected threshold = **$0.42$** (OOF F1 at 0.50 = 0.6864 $\to$ OOF F1 at 0.42 = **0.7004**)
- **Methodological Distinction (Critical):**
  - The maximum threshold-selected OOF F1 score (e.g. 0.7025 for RF) is a **SELECTION METRIC**, NOT an unbiased estimate of generalization.
  - Because the threshold is selected to maximize F1 on the exact same OOF predictions, evaluating on those predictions incurs post-hoc selection optimism.
  - The true independent generalization performance of the selected threshold is measured strictly on the locked holdout test partition:
    - Holdout RF F1 at $\theta = 0.48$: **0.6822** (Recall: 0.8148, Precision: 0.5867, ROC-AUC: 0.8333)
    - Holdout XGBoost F1 at $\theta = 0.42$: **0.6970** (Recall: 0.8519, Precision: 0.5897, ROC-AUC: 0.8404)
- **Holdout Independence:** Holdout labels and probabilities were completely excluded from the threshold search.

---

## 3. Verify Holdout Isolation

**Status:** **PASS**

### Audit & Evidence
- **Split Logic:** In [`ml/model_utils.py`](ml/model_utils.py#L55-L100), `split_data()` performs:
  `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`
- **Partition Sizes:**
  - Training partition: Exactly **615 records** (400 Class 0, 215 Class 1)
  - Holdout test partition: Exactly **154 records** (100 Class 0, 54 Class 1)
- **Imputation Partitioning:**
  - Imputation medians are calculated strictly on `X_train` (`Glucose`: 117.0, `BloodPressure`: 72.0, `SkinThickness`: 29.0, `Insulin`: 125.0, `BMI`: 32.4).
  - During 5-fold cross-validation (`cross_validate_models_leak_free`), medians are re-computed fold-by-fold strictly inside `X_tr` of each training fold; validation folds and holdout samples never enter imputer fitting.
- **Holdout Lifecycle & Consumption Status:**
  - The holdout partition was evaluated exactly once during the previous final execution of `ml/run_pipeline.py`.
  - **In accordance with standard protocol, the holdout partition is now treated as CONSUMED.** It was not re-visited or rerun during this verification phase.
- Verified via automated tests: `test_01_holdout_split_preservation` and `test_02_leak_free_imputation` both pass.

---

## 4. Verify Actual Application Inference

**Status:** **FAIL** (Static Web View) / **PASS** (Python Backend Engine)

### Detailed Trace from User Input to Output

```
[Path A: Static Frontend Interface (Current Web View)]
User enters values in HTML inputs (app.py -> st.iframe -> frontend/index.html)
        ↓
JavaScript calculateRisk() in build_frontend.py (lines 1935-1976)
        ↓
Calculates risk percentage via hardcoded client-side linear factors
(e.g., glucFactor * (gluc - med_gluc) + bmiFactor * (bmi - med_bmi) + baseVal)
        ↓
Displays risk percentage & risk tier in DOM
[Result: Does NOT invoke Python, models/best_model.pkl, or sklearn predict_proba()]
```

```
[Path B: True Python Inference Service (explainability/shap_analysis.py)]
Input patient dictionary
        ↓
DiabetesExplainerService.explain_prediction()
        ↓
Validates 8 required features in strict order (FEATURE_COLUMNS)
        ↓
Imputes missing/zero values with training medians from models/metadata.json
        ↓
Invokes trained model: self.model.predict_proba(df_input)[0, 1]
        ↓
Applies saved optimal decision threshold (self.threshold from metadata.json)
        ↓
Executes shap.TreeExplainer to compute exact TreeSHAP values and waterfall explanation
[Result: Verified, leak-free, model-driven inference]
```

### Verified Discrepancy & Evidence
1. **The Web UI is Sandboxed in an Iframe:**  
   In [`app.py`](app.py#L48-L62), Streamlit renders `frontend/index.html` inside `st.iframe(html_markup, height=2200)`. Because this is a static client-side document running in an iframe sandbox without a live WebSocket or REST API bridge back to the Python process, user inputs never trigger Python functions.
2. **The Frontend Uses Client-Side Logic:**  
   In [`build_frontend.py`](build_frontend.py#L1935-L1976), `calculateRisk()` calculates risk numbers using client-side JavaScript math rather than calling `models/best_model.pkl`.
3. **Alignment Fix Applied:**  
   We performed a minimal correction to [`build_frontend.py`](build_frontend.py#L1940-L1978) ensuring that the client-side JavaScript strictly uses the training-set medians (`117.0`, `32.4`, `72.0`, `29.0`, `125.0`), the saved model decision thresholds (`0.48` for RF, `0.42` for XGBoost), and holdout AUCs from [`models/metadata.json`](models/metadata.json).
4. **Backend Readiness:**  
   The true Python service `explain_prediction()` in `explainability/shap_analysis.py` is fully functional and passes automated test `test_09_explainer_service_workflow`.

---

## 5. Verify SHAP Consistency

**Status:** **PASS**

### Audit & Evidence
- **Model Verification:**  
  [`explainability/run_shap.py`](explainability/run_shap.py#L60-L70) loads `models/best_model.pkl`. It successfully loaded `RandomForestClassifier` (the selected best model).
- **Feature Schema & Preprocessing:**  
  Feature ordering strictly enforces `['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']`. Imputation medians are loaded directly from `models/metadata.json` (fitted only on training data).
- **Positive-Class Attribution Handling:**  
  Random Forest produces a 3-D SHAP explanation matrix of shape $(N, \text{features}, 2)$. In [`explainability/shap_analysis.py`](explainability/shap_analysis.py#L84-L95), `compute_shap_values()` explicitly isolates the positive class (Class 1 = Diabetic):
  ```python
  if hasattr(explanation, "values") and explanation.values.ndim == 3:
      explanation = explanation[:, :, 1]
  ```
  And in `DiabetesExplainerService.explain_prediction()`:
  ```python
  if np.ndim(raw_shap) == 2:
      shap_values = raw_shap[:, 1]
      base_value = float(raw_base[1])
  ```
- **Generated SHAP Artifacts:**  
  `results/shap_summary_beeswarm.png`, `results/shap_summary_bar.png`, `results/shap_dependence_*.png`, waterfall plots, and [`results/shap_interpretation.md`](results/shap_interpretation.md) were generated and verified.

---

## 6. Verify Tests and Artifacts

**Status:** **PASS**

### Audit & Evidence
- **Automated Test Suite:**  
  Executed `python -m pytest tests/test_pipeline.py -v` without re-running training or accessing the holdout. All 10 tests passed in 7.22s:
  - `test_01_holdout_split_preservation`: **PASSED** (154 holdout records preserved)
  - `test_02_leak_free_imputation`: **PASSED** (test set never enters imputer fitting)
  - `test_03_out_of_fold_coverage`: **PASSED** (all 615 training records covered by OOF)
  - `test_04_threshold_selection_isolation`: **PASSED** (thresholds tuned on OOF only)
  - `test_05_metadata_and_threshold_used_in_inference`: **PASSED** (service uses metadata threshold)
  - `test_06_feature_consistency`: **PASSED** (8-feature ordering matches across modules)
  - `test_07_both_models_available`: **PASSED** (RF, XGBoost, best_model loadable)
  - `test_08_model_selection_uses_validation_metrics`: **PASSED** (winner chosen by CV metrics)
  - `test_09_explainer_service_workflow`: **PASSED** (end-to-end Python inference & SHAP)
  - `test_10_evaluation_metric_computation`: **PASSED** (metrics in valid non-zero ranges)

- **Artifact Integrity:**
  - Model checkpoints exist and are valid: `models/best_model.pkl` (575 KB), `models/best_model.joblib` (575 KB), `models/random_forest.pkl` (1.07 MB), `models/random_forest.joblib` (1.07 MB), `models/xgboost.pkl` (575 KB), `models/xgboost.joblib` (575 KB).
  - Metadata file [`models/metadata.json`](models/metadata.json) records model parameters, training medians, package versions, and optimal decision thresholds.
  - Tabular results in `results/` match holdout metrics:
    - [`results/metrics.csv`](results/metrics.csv): RF F1 = 0.6822, XGBoost F1 = 0.6970
    - [`results/cross_validation_results.csv`](results/cross_validation_results.csv): RF CV F1 = 0.7001 ± 0.0502, XGBoost CV F1 = 0.6851 ± 0.0368

---

## 7. Discrepancies in Documentation

During the audit, the following documentation discrepancies were verified in [`README.md`](README.md):

1. **Model Selection Protocol Description (Stale):**  
   Lines 136–138 of `README.md` state:  
   *"Best model selected based on F1-score on untouched test set. XGBoost was selected as best model."*  
   **Correction:** This reflected the pre-audit legacy code that had test leakage. The pipeline now selects the winner strictly from training-side 5-fold cross-validation, where **Random Forest** was selected as the best model.
2. **Inference Flow Description (Stale):**  
   Lines 226–229 of `README.md` depict `User Input` flowing into `Saved XGBClassifier (best_model.pkl)`. In reality, the web application UI in `app.py` renders a static compiled HTML page.

---

## 8. Final Readiness Recommendation

### Readiness Assessment: **PRODUCTION-READY FOR BENCHMARKING & EXPERIMENTATION; CONDITIONAL FOR CLINICAL UI**

1. **Core ML & Explainability Pipeline:** **APPROVED.**  
   Data splitting, leak-free median imputation, class imbalance weighting, out-of-fold threshold optimization, model selection, SHAP TreeExplainer, and the automated test suite strictly adhere to all scientific and leakage-safe constraints.
2. **Application Inference:** **CONDITIONAL.**  
   The application UI currently functions as an interactive demonstration/dashboard using compiled assets. For regulatory or production clinical deployment requiring real-time scikit-learn model inference from user keystrokes, the frontend form should be wired to `DiabetesExplainerService.explain_prediction()` via a Streamlit native form or bidirectional REST component.
