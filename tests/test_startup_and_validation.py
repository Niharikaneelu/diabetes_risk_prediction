"""
Unit and Integration Tests for Startup UI Error Fixes:
1. Streamlit startup indicator suppression and server readiness
2. False validation error suppression on page load & model switching
3. Preserved validation on explicit empty/missing submissions
4. Frontend build consistency via build_frontend.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def frontend_html() -> str:
    path = PROJECT_ROOT / "frontend" / "index.html"
    assert path.exists(), "frontend/index.html does not exist"
    return path.read_text(encoding="utf-8")


def test_01_app_cache_resource_show_spinner_false():
    """Verify that app.py uses @st.cache_resource(show_spinner=False)."""
    app_py_content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "@st.cache_resource(show_spinner=False)" in app_py_content, (
        "app.py must configure @st.cache_resource with show_spinner=False to suppress top-right startup indicator"
    )


def test_02_server_readiness_probe_implemented():
    """Verify that ml/server.py verifies readiness before returning the port."""
    server_py_content = (PROJECT_ROOT / "ml" / "server.py").read_text(encoding="utf-8")
    assert "/health" in server_py_content, "ml/server.py must probe /health to verify readiness"
    assert "urllib.request" in server_py_content


def test_03_default_model_initialization_on_dom_content_loaded(frontend_html: str):
    """Verify DOMContentLoaded calls initializeDefaultModel('random_forest') instead of bare changeModel."""
    assert "initializeDefaultModel('random_forest');" in frontend_html
    assert "function initializeDefaultModel(modelName)" in frontend_html
    # Ensure changeModel('random_forest') is NOT directly executed inside DOMContentLoaded listener
    dom_content_loaded_match = re.search(r"window\.addEventListener\('DOMContentLoaded', \(\) => \{(.*?)\}\);", frontend_html, re.DOTALL)
    assert dom_content_loaded_match is not None
    dom_body = dom_content_loaded_match.group(1)
    assert "initializeDefaultModel('random_forest');" in dom_body
    assert "calculateRisk(true);" not in dom_body


def test_04_change_model_does_not_calculate_on_empty_form(frontend_html: str):
    """Verify changeModel only invokes calculateRisk if validateFormInputs({ notify: false }) is true."""
    # Find changeModel function
    change_model_match = re.search(r"function changeModel\(modelName, options = \{\}\) \{(.*?)\n    \}", frontend_html, re.DOTALL)
    assert change_model_match is not None, "changeModel must accept options parameter"
    fn_body = change_model_match.group(1)

    assert "if (validateFormInputs({ notify: false }))" in fn_body
    assert "calculateRisk({ notifyValidation: false });" in fn_body
    # Ensure unconditional calculateRisk is absent
    assert "\n      calculateRisk(true);" not in fn_body


def test_05_validation_notification_options_supported(frontend_html: str):
    """Verify validateFormInputs and calculateRisk support silent notification options."""
    # validateFormInputs supports options.notify
    val_match = re.search(r"function validateFormInputs\(options = \{\}\) \{(.*?)\n    \}", frontend_html, re.DOTALL)
    assert val_match is not None
    val_body = val_match.group(1)
    assert "const notify = (options && typeof options === 'object' && options.notify !== undefined) ? options.notify : true;" in val_body
    assert "if (notify) {" in val_body

    # calculateRisk supports options.notifyValidation
    calc_match = re.search(r"async function calculateRisk\(options = \{\}\) \{(.*?)\n    \}", frontend_html, re.DOTALL)
    assert calc_match is not None
    calc_body = calc_match.group(1)
    assert "notifyValidation" in calc_body
    assert "if (!validateFormInputs({ notify: notifyValidation }))" in calc_body


def test_06_form_submission_preserves_active_validation(frontend_html: str):
    """Verify clicking Estimate Diabetes Risk or form submit still validates actively."""
    assert 'onclick="calculateRisk()"' in frontend_html
    assert 'onsubmit="event.preventDefault(); calculateRisk();"' in frontend_html


def test_07_build_consistency_survives_rebuild():
    """Verify build_frontend.py generates the exact required startup logic."""
    res = subprocess.run([sys.executable, str(PROJECT_ROOT / "build_frontend.py")], capture_output=True, text=True)
    assert res.returncode == 0, f"build_frontend.py failed with: {res.stderr}"

    rebuilt_html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "initializeDefaultModel('random_forest');" in rebuilt_html
    assert "if (validateFormInputs({ notify: false }))" in rebuilt_html
    assert "calculateRisk({ notifyValidation: false });" in rebuilt_html
