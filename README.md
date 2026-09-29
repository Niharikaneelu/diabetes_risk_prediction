# Diabetes Risk Prediction

A complete, reproducible machine-learning project for diabetes risk classification with the Pima Indians Diabetes dataset, featuring model training, evaluation, SHAP explainability, and an interactive Streamlit application.

## Project layout

```
diabetes_risk_prediction/
├── data/
│   └── diabetes.csv              # Pima Indians Diabetes dataset (768 patients, 8 features)
├── ml/
│   ├── run_pipeline.py           # Phase 2 end-to-end training pipeline
│   ├── train_models.py           # Random Forest & XGBoost training + hyperparameter tuning
│   ├── evaluate_models.py        # Evaluation metrics, ROC, confusion matrix
│   └── model_utils.py            # Data loading, split, visualization utilities
├── explainability/
│   ├── run_shap.py               # Phase 3 global SHAP analysis execution script
│   ├── shap_analysis.py          # SHAP engine: TreeExplainer, plots, inference service
│   └── __init__.py               # Package exports
├── src/
│   ├── preprocessing.py          # Feature definitions and preprocessing helpers
│   ├── models.py                 # Model construction utilities
│   └── evaluation.py             # Evaluation helper stubs
├── models/
│   ├── best_model.pkl            # Best model (XGBoost, selected by Phase 2)
│   ├── best_model.joblib         # Same model in joblib format
│   ├── xgboost.pkl / .joblib     # Tuned XGBoost model
│   └── random_forest.pkl / .joblib  # Tuned Random Forest model
├── results/
│   ├── metrics.csv               # Test set performance metrics
│   ├── model_comparison.csv      # Side-by-side model comparison
│   ├── shap_feature_importance.csv  # Global SHAP importance rankings
│   ├── shap_summary_bar.png      # SHAP bar plot
│   ├── shap_summary_beeswarm.png # SHAP beeswarm plot
│   ├── shap_interpretation.md    # Automated SHAP interpretation report
│   └── individual_explanations/  # Waterfall plots for sample patients
├── notebooks/
│   ├── 01_data_analysis.ipynb    # Exploratory data analysis
│   ├── 02_model_training.ipynb   # Model training walkthrough
│   └── 03_shap_analysis.ipynb    # SHAP explainability walkthrough
├── frontend/
│   └── index.html                # Compiled clinical decision support UI
├── build_frontend.py             # Healthcare UI asset compiler
├── app.py                        # Phase 3 interactive Streamlit application
├── requirements.txt              # Python package dependencies
└── .gitignore                    # Git ignore specifications
```

## Setup

### 1. Clone repository & create virtual environment

**On Windows (PowerShell):**
```powershell
cd diabetes_risk_prediction
python -m venv .venv

# If script execution is restricted in PowerShell, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
cd diabetes_risk_prediction
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
cd diabetes_risk_prediction
python3 -m venv .venv
source .venv/bin/activate

# macOS users: XGBoost requires OpenMP
brew install libomp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipelines

### 1. Run the ML training pipeline (Phase 2):
```bash
python ml/run_pipeline.py
```
> Trains Random Forest and XGBoost with 5-fold cross-validation, selects the best model, and outputs performance plots and model weights into `models/` and `results/`.

### 2. Run the SHAP explainability pipeline (Phase 3):
```bash
python explainability/run_shap.py
```
> Computes TreeExplainer SHAP values on test data, generates global beeswarm and dependence plots, individual waterfall charts, and `results/shap_interpretation.md`.

### 3. Launch the Streamlit application:
```bash
# Direct command (if Streamlit is in PATH or venv is activated):
streamlit run app.py

# Alternatively, run via Python module (recommended on Windows to avoid PATH issues):
python -m streamlit run app.py
```

### 4. Run the interactive notebooks:
```bash
jupyter notebook
```

---

## Phase 1 — Data Processing

The Pima Indians Diabetes dataset contains 768 patient records with 8 clinical features and a binary outcome (diabetic/non-diabetic).

**Preprocessing:**
- Zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as physiologically impossible and replaced with column medians
- Strict train/test split (80/20, stratified) before any imputation to prevent data leakage
- Imputation medians computed exclusively from the training set

---

## Phase 2 — ML Model Training and Evaluation

**Models trained:**
- Random Forest (tuned via 5-fold GridSearchCV on F1-score)
- XGBoost (tuned via 5-fold GridSearchCV on F1-score)

**Model selection:**
- Best model selected based on F1-score on untouched test set
- XGBoost was selected as best model and saved to `models/best_model.pkl`

**Feature order used during training (and required at inference):**
1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

> Note: No StandardScaler was applied in the final training pipeline. Raw imputed values are fed directly to the tree-based models.

---

## Phase 3 — Explainability + Application

### Why SHAP?

SHAP (SHapley Additive exPlanations) provides mathematically rigorous, model-agnostic feature attributions based on game theory (Shapley values). Unlike feature importance scores from tree splits, SHAP:

- Is **consistent** — more important features always get higher SHAP values
- Is **locally faithful** — SHAP values sum exactly to the prediction minus the baseline
- Is **globally interpretable** — mean absolute SHAP gives a faithful global importance ranking
- Provides **directionality** — SHAP values show whether a feature increases or decreases risk

For tree-based models (XGBoost, Random Forest), `TreeExplainer` computes exact SHAP values efficiently in polynomial time.

### Global Explainability

Run the global analysis to generate:

```bash
python explainability/run_shap.py
```

This generates:
- `results/shap_summary_bar.png` — Feature importance by mean |SHAP value|
- `results/shap_summary_beeswarm.png` — SHAP beeswarm distribution plot
- `results/shap_dependence_*.png` — Feature effect plots for top predictors
- `results/shap_feature_importance.csv` — Ranked feature importances
- `results/feature_importance_comparison.csv` — XGBoost native vs SHAP comparison
- `results/individual_explanations/` — Sample patient waterfall plots
- `results/shap_interpretation.md` — Automated interpretation report

**Key global findings (from test set, N=154):**

| Rank | Feature | Mean |SHAP Value| |
|------|---------|-------------------|
| 1 | Glucose | 0.9319 |
| 2 | BMI | 0.5229 |
| 3 | Age | 0.2772 |
| 4 | DiabetesPedigreeFunction | 0.2310 |
| 5 | Pregnancies | 0.1522 |
| 6 | Insulin | 0.1344 |
| 7 | SkinThickness | 0.0548 |
| 8 | BloodPressure | 0.0416 |

> Glucose is the dominant predictor, followed by BMI and Age. SHAP ranks agree closely with XGBoost's native feature importance (Spearman rank correlation: all top-3 features agree).

### SHAP Waterfall Plots

Each individual prediction is explained by a waterfall plot showing:
- Base value `E[f(x)]` — average model output across all training data
- Per-feature SHAP contributions (red = increases risk, blue = decreases risk)
- Final predicted value `f(x)` = base + sum(SHAP values)

### Local (Patient-Specific) Explainability

For every prediction made through the application, the system:

1. Accepts patient input (8 features)
2. Applies the same preprocessing used during training (zero → median imputation)
3. Passes through the saved XGBoost model
4. Computes exact SHAP values using `TreeExplainer`
5. Identifies top risk-increasing and risk-decreasing features
6. Renders an interactive waterfall plot

### Model Inference Flow

```
User Input (8 features)
        ↓
Zero → Median Imputation (using training-set medians)
        ↓
Feature Ordering Enforced
        ↓
Saved XGBClassifier (best_model.pkl)
        ↓
predict_proba() → Risk Probability
predict() → Class Label (0 = Low Risk, 1 = High Risk)
        ↓
SHAP TreeExplainer(model)(df_input) → Explanation object
        ↓
Local SHAP Waterfall + Feature Contributions Table
```

### Streamlit Application

The application provides a polished, professional interface with:

- **Patient input form** — 8 clinical measurement inputs with units and validation ranges
- **Risk probability card** — Large visual probability display with color-coded result
- **Top features summary** — Risk-increasing vs risk-decreasing features at a glance
- **Feature Contributions tab** — Sortable table of SHAP contributions per feature
- **Patient SHAP Explanation tab** — Waterfall plot for the specific prediction
- **Global Model Insights tab** — Bar plot, beeswarm plot, and feature importance table
- **Model selection sidebar** — Switch between Best Model, Random Forest, and XGBoost

### Architecture

```
                    ┌─────────────────┐
                    │   User Input    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Preprocessing  │
                    │ (zero→median)   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Trained XGBoost │
                    │      Model      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Prediction    │
                    │ Probability +   │
                    │ Classification  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │      SHAP       │
                    │   TreeExplainer │
                    └────────┬────────┘
                             ↓
             ┌───────────────┴───────────────┐
             ↓                               ↓
     Local Explanation                Global Explanation
             ↓                               ↓
     Waterfall Plot                  Beeswarm / Bar Plot
             │                               │
             └───────────────┬───────────────┘
                             ↓
                    Streamlit Dashboard
```

### Important Assumptions and Limitations

1. **No scaler in inference path** — The saved XGBoost model was trained on raw imputed values (no StandardScaler). The application uses the same raw preprocessing.
2. **Imputation medians** — Computed from training set only. At inference, zero/null values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are replaced with training-set medians.
3. **Dataset-specific** — The model was trained on the Pima Indians Diabetes dataset, which has specific demographic characteristics. Predictions should not be generalized to all populations.
4. **Educational use only** — This is NOT a medical diagnostic tool. SHAP values describe model behavior, not biological causation.

---

## Disclaimer

This application is intended for **educational and research purposes only**. It is not a medical diagnostic tool. Predictions should not replace professional medical assessment. Always consult a licensed healthcare provider.
