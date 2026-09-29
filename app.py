"""
Diabetes Risk Prediction & Explainability — Streamlit Application
Phase 3: Model Explainability + Modern Healthcare UI
"""

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk AI • Clinical Decision Support",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit chrome, paddings, and borders for seamless UI presentation
st.markdown("""
<style>
    #MainMenu, header, footer { display: none !important; visibility: hidden !important; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    .stApp {
        background-color: #faf8ff !important;
    }
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        z-index: 999999 !important;
    }
</style>
""", unsafe_allow_html=True)

frontend_file = PROJECT_ROOT / "frontend" / "index.html"
if not frontend_file.exists():
    import subprocess
    subprocess.run([sys.executable, str(PROJECT_ROOT / "build_frontend.py")], check=True)

with open(frontend_file, "r", encoding="utf-8") as f:
    html_markup = f.read()

# Render complete healthcare AI application UI
components.html(html_markup, height=2200, scrolling=True)
