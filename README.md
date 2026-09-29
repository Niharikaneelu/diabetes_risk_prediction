# Diabetes Risk Prediction

A small, reproducible machine-learning project for exploring diabetes risk classification with the Pima Indians Diabetes dataset format.

## Project layout

- `data/diabetes.csv`: input data with the eight predictor columns and `Outcome` target.
- `ml/`: dedicated machine learning pipeline (`train_models.py`, `evaluate_models.py`, `model_utils.py`, `run_pipeline.py`).
- `explainability/`: dedicated SHAP explainability pipeline (`shap_analysis.py`, `run_shap.py`).
- `notebooks/`: analysis, model training, and SHAP explainability workflows (`01_data_analysis.ipynb`, `02_model_training.ipynb`, `03_shap_analysis.ipynb`).
- `src/`: reusable preprocessing, model, evaluation, and explainability helpers.
- `models/`: generated trained model artifacts (`best_model.pkl`, `random_forest.pkl`, `xgboost.pkl`, `.joblib`).
- `results/`: evaluation tables, confusion matrices, ROC curves, feature importances, SHAP beeswarm/waterfall plots, and interpretation report.
- `app.py`: interactive Streamlit prediction and explainability interface.

## Setup

```powershell
cd diabetes-risk-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the ML pipeline:

```powershell
python ml/run_pipeline.py
```

Run the SHAP explainability pipeline:

```powershell
python explainability/run_shap.py
```

Run the notebooks with:

```powershell
jupyter notebook
```

Run the app after training models:

```powershell
streamlit run app.py
```

The application is for educational use and is not a medical diagnostic tool.
