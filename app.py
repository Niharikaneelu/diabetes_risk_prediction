"""Streamlit app for interactive diabetes risk estimation."""

from pathlib import Path

import streamlit as st

from src.models import get_models
from src.preprocessing import FEATURE_COLUMNS

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="+", layout="centered")
st.title("Diabetes Risk Prediction")
st.caption("Educational screening support only; this is not a medical diagnosis.")

st.subheader("Patient measurements")
values = {}
for feature in FEATURE_COLUMNS:
    if feature == "Pregnancies":
        values[feature] = st.number_input(feature, min_value=0, max_value=20, value=1, step=1)
    elif feature == "Age":
        values[feature] = st.number_input(feature, min_value=1, max_value=120, value=30, step=1)
    elif feature == "DiabetesPedigreeFunction":
        values[feature] = st.number_input(feature, min_value=0.0, max_value=3.0, value=0.47, step=0.01)
    else:
        values[feature] = st.number_input(feature, min_value=0.0, value=100.0, step=1.0)

if st.button("Estimate risk", type="primary"):
    model_path = Path(__file__).parent / "models" / "random_forest.joblib"
    if not model_path.exists():
        st.warning("No trained model found. Run notebook 02_model_training.ipynb first.")
    else:
        from src.models import load_model
        import pandas as pd

        model = load_model(model_path)
        row = pd.DataFrame([values])
        probability = model.predict_proba(row)[0, 1]
        st.metric("Estimated probability", f"{probability:.1%}")
        if probability >= 0.5:
            st.error("Higher predicted risk. Please discuss results with a qualified clinician.")
        else:
            st.success("Lower predicted risk. Continue routine health monitoring.")
