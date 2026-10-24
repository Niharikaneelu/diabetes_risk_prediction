"""
Diabetes Risk Prediction & Explainability — Streamlit Application
Phase 4: Simplified & Mathematically Verified Explainability UI (v2)
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk AI",
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

# Start background inference server
@st.cache_resource(show_spinner=False)
def get_prediction_server_port() -> int:
    from ml.server import start_prediction_server
    return start_prediction_server(initial_port=8502)

api_port = get_prediction_server_port()

frontend_file = PROJECT_ROOT / "frontend" / "index.html"
if not frontend_file.exists():
    import subprocess
    subprocess.run([sys.executable, str(PROJECT_ROOT / "build_frontend.py")], check=True)

with open(frontend_file, "r", encoding="utf-8") as f:
    html_markup = f.read()

# Inject active API port and optional page param into the page context
page_param = st.query_params.get("page", "")
port_script = f"<script>window.PREDICTION_API_PORT = {api_port}; window.INITIAL_PAGE = '{page_param}';</script>"
if "<head>" in html_markup:
    html_markup = html_markup.replace("<head>", f"<head>\n  {port_script}", 1)
else:
    html_markup = f"{port_script}\n{html_markup}"

# Render complete healthcare AI application UI using st.iframe
if hasattr(st, "iframe"):
    st.iframe(html_markup, height=2200)
else:
    import streamlit.components.v1 as components
    components.html(html_markup, height=2200, scrolling=True)
