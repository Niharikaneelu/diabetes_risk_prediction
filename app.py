"""Streamlit app for interactive diabetes risk estimation."""

from pathlib import Path

import streamlit as st

from src.models import get_models
from src.preprocessing import FEATURE_COLUMNS

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="+", layout="centered")
st.title("Diabetes Risk Prediction")
st.caption("Educational screening support only; this is not a medical diagnosis.")

st.subheader("Model Selection")
model_choice = st.selectbox(
    "Choose classifier:",
    options=["Best Model (XGBoost)", "Random Forest", "XGBoost"],
    index=0,
)
model_file_map = {
    "Best Model (XGBoost)": "best_model.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}

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
    chosen_file = model_file_map.get(model_choice, "best_model.joblib")
    model_path = Path(__file__).parent / "models" / chosen_file
    if not model_path.exists():
        st.warning(f"No trained model found at {model_path}. Run python ml/run_pipeline.py first.")
    else:
        from explainability import explain_prediction
        import matplotlib.pyplot as plt
        import pandas as pd
        import shap

        res = explain_prediction(values)
        prob = res["probability"]
        pred_label = res["prediction_label"]

        st.subheader("Prediction Result")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Estimated Risk Probability", f"{prob:.1%}")
        with col_m2:
            if prob >= 0.5:
                st.error(f"Prediction: {pred_label}")
            else:
                st.success(f"Prediction: {pred_label}")

        st.markdown("---")
        st.subheader("SHAP Prediction Explanation")
        st.write(
            "The model output is explained by quantifying how each clinical measurement pushes "
            "the predicted risk above or below the baseline prior."
        )

        # Tabular contribution breakdown
        contrib_data = []
        for feat, info in res["feature_contributions"].items():
            contrib_data.append({
                "Feature": feat,
                "Value": f"{info['feature_value']:.2f}",
                "SHAP Contribution (Log-Odds)": f"{info['shap_value']:+.4f}",
                "Impact Direction": info["direction"],
            })
        df_contrib = pd.DataFrame(contrib_data)
        # Sort by absolute SHAP contribution descending
        df_contrib["_abs"] = [abs(info["shap_value"]) for info in res["feature_contributions"].values()]
        df_contrib = df_contrib.sort_values(by="_abs", ascending=False).drop(columns=["_abs"])
        st.dataframe(df_contrib, use_container_width=True, hide_index=True)

        # Patient-specific SHAP Waterfall Plot
        st.subheader("Patient Risk Contribution (SHAP Waterfall Plot)")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        shap.plots.waterfall(res["explanation"], show=False)
        plt.title(f"Patient Explanation (Probability: {prob:.1%})", fontsize=11, weight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Medical Disclaimer: This system is intended for educational/research purposes "
            "and should not be used as a substitute for professional medical diagnosis. "
            "Always consult a licensed medical clinician for clinical assessment."
        )
