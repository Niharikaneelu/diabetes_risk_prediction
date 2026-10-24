"""
Lightweight REST Inference Server for Diabetes Risk Prediction.

Serves model predictions and SHAP explanations from trained models
(Random Forest and XGBoost) to the web interface with full CORS support.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from explainability.shap_analysis import DiabetesExplainerService
from ml.model_utils import FEATURE_COLUMNS, load_metadata

logger = logging.getLogger("diabetes_server")

_SERVER_INSTANCE: Optional[ThreadingHTTPServer] = None
_SERVER_PORT: Optional[int] = None
_SERVER_LOCK = threading.Lock()


class PredictionAPIHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler for model inference and health checks.
    """

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard access logging to keep console clean
        pass

    def _set_cors_headers(self, status_code: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_cors_headers(200)

    def do_GET(self) -> None:
        if self.path in ("/health", "/api/health"):
            self._set_cors_headers(200)
            meta = load_metadata()
            best_model = meta.get("best_model_name", "Random Forest")
            response = {
                "status": "healthy",
                "service": "Diabetes Risk ML Engine",
                "best_model": best_model,
                "supported_models": ["Random Forest", "XGBoost"],
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        elif self.path in ("/metadata", "/api/metadata"):
            self._set_cors_headers(200)
            meta = load_metadata()
            self.wfile.write(json.dumps(meta).encode("utf-8"))
        else:
            self._set_cors_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self) -> None:
        if self.path in ("/api/predict", "/predict"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self._set_cors_headers(400)
                    self.wfile.write(json.dumps({"error": "Empty request body"}).encode("utf-8"))
                    return

                body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(body)

                model_name = payload.get("model", "best")
                features = payload.get("features", {})

                if not isinstance(features, dict):
                    self._set_cors_headers(400)
                    self.wfile.write(json.dumps({"error": "'features' must be an object/dict"}).encode("utf-8"))
                    return

                # Get model explainer service
                service = DiabetesExplainerService.get_instance(model_name=model_name)
                result = service.explain_prediction(features)

                # Format clean serializable response
                is_high = float(result["probability"]) >= float(result["threshold"])
                tier = "elevated" if is_high else "low"
                label = f"High Risk (Diabetic • Th: {result['threshold']:.2f})" if is_high else f"Low Risk (Non-Diabetic • Th: {result['threshold']:.2f})"

                clean_contributions = {}
                for feat, info in result.get("feature_contributions", {}).items():
                    clean_contributions[feat] = {
                        "val": float(info["feature_value"]),
                        "shap": float(info["shap_value"]),
                        "feature_value": float(info["feature_value"]),
                        "shap_value": float(info["shap_value"]),
                        "direction": info["direction"],
                    }

                clean_response = {
                    "status": "success",
                    "model_name": service.model_name,
                    "model_used": service.model_name,
                    "probability": float(result["probability"]),
                    "predicted_class": int(result["predicted_class"]),
                    "prediction_label": result["prediction_label"],
                    "threshold": float(result["threshold"]),
                    "base_value": float(result["base_value"]),
                    "risk_tier": tier,
                    "risk_label": label,
                    "shap_contributions": clean_contributions,
                    "feature_contributions": clean_contributions,
                    "top_positive": [(f, float(s), float(v)) for f, s, v in result.get("top_positive_contributors", [])],
                    "top_negative": [(f, float(s), float(v)) for f, s, v in result.get("top_negative_contributors", [])],
                }

                self._set_cors_headers(200)
                self.wfile.write(json.dumps(clean_response).encode("utf-8"))

            except Exception as e:
                self._set_cors_headers(500)
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "message": "Failed to generate prediction from model artifact",
                }
                self.wfile.write(json.dumps(error_response).encode("utf-8"))
        else:
            self._set_cors_headers(404)
            self.wfile.write(json.dumps({"error": "Unknown POST endpoint"}).encode("utf-8"))


def start_prediction_server(
    host: str = "0.0.0.0",
    initial_port: int = 8502,
    max_retries: int = 5,
) -> int:
    """
    Start the prediction HTTP server in a background daemon thread.
    Returns the active bound port.
    """
    global _SERVER_INSTANCE, _SERVER_PORT

    with _SERVER_LOCK:
        if _SERVER_INSTANCE is not None and _SERVER_PORT is not None:
            return _SERVER_PORT

        for port in range(initial_port, initial_port + max_retries):
            try:
                server = ThreadingHTTPServer((host, port), PredictionAPIHandler)
                _SERVER_INSTANCE = server
                _SERVER_PORT = port

                thread = threading.Thread(target=server.serve_forever, daemon=True, name="PredictionServerThread")
                thread.start()

                # Verify server readiness via health probe
                import time
                import urllib.request
                for _ in range(20):
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as resp:
                            if resp.status == 200:
                                break
                    except Exception:
                        time.sleep(0.05)

                return port
            except OSError:
                continue

        raise RuntimeError(f"Could not bind Prediction Server to ports {initial_port}–{initial_port + max_retries}")


def get_active_server_port() -> Optional[int]:
    """Return the active server port if started, or None."""
    return _SERVER_PORT
