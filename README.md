# Diabetes Risk Prediction

A small, reproducible machine-learning project for exploring diabetes risk classification with the Pima Indians Diabetes dataset format.

## Project layout

- `data/diabetes.csv`: input data with the eight predictor columns and `Outcome` target.
- `notebooks/`: analysis, model training, and SHAP explainability workflows.
- `src/`: reusable preprocessing, model, evaluation, and explainability code.
- `models/`: generated model artifacts.
- `app.py`: optional Streamlit prediction interface.

## Setup

```powershell
cd diabetes-risk-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the notebooks with:

```powershell
jupyter notebook
```

Run the app after training a model:

```powershell
streamlit run app.py
```

The application is for educational use and is not a medical diagnostic tool.
