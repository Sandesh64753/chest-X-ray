import os
import io
import sys
import psutil
import gc
import pytest
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from app import app, analyze_single_xray
import config

client = TestClient(app)

@pytest.fixture(scope="session")
def valid_png_bytes():
    """Generates a small valid synthetic 224x224 RGB image in PNG bytes."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Case 1: Homepage returns HTTP 200
def test_01_homepage_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "ChestVision AI" in response.text
    assert "Understand Your Chest X-ray" in response.text


# Case 2: Health and readiness endpoints return expected responses
def test_02_health_endpoint():
    # Health endpoint
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "models_loaded" in data

    # Readiness endpoint
    resp_ready = client.get("/ready")
    if data["models_loaded"]:
        assert resp_ready.status_code == 200
        assert resp_ready.json()["status"] == "ready"
    else:
        assert resp_ready.status_code == 503


# Case 3: Valid X-ray upload completes successfully
def test_03_valid_xray_upload(valid_png_bytes):
    response = client.post(
        "/analyze",
        files={"file": ("test_xray.png", valid_png_bytes, "image/png")}
    )
    assert response.status_code == 200
    assert "Analysis Dashboard" in response.text
    assert "Classification Result" in response.text
    assert "Lung Segmentation" in response.text
    assert "Why did the model make this prediction?" in response.text


# Case 4: Empty upload is rejected appropriately
def test_04_empty_upload_rejected():
    response = client.post(
        "/analyze",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400
    assert "empty" in response.text.lower()


# Case 5: Invalid image is rejected appropriately
def test_05_invalid_image_rejected():
    response = client.post(
        "/analyze",
        files={"file": ("corrupted.png", b"This is plainly not a PNG image file", "image/png")}
    )
    assert response.status_code == 400
    assert ("not a valid" in response.text.lower() or "error" in response.text.lower())


# Case 6: Oversized image is rejected appropriately
def test_06_oversized_image_rejected(monkeypatch):
    # Temporarily set max upload to 500 bytes for test
    monkeypatch.setattr(app_module, "MAX_UPLOAD_SIZE_BYTES", 500)
    oversized_data = b"0" * 1000
    response = client.post(
        "/analyze",
        files={"file": ("large.png", oversized_data, "image/png")}
    )
    assert response.status_code in [400, 413]
    assert ("exceeds" in response.text.lower() or "too large" in response.text.lower())


# Case 7: Missing model produces a clear, diagnosable failure
def test_07_missing_model_diagnosable(monkeypatch, valid_png_bytes):
    monkeypatch.setattr(app_module, "models_loaded", False)
    monkeypatch.setattr(app_module, "model_error_msg", "Test model missing error")

    # Ready endpoint should return 503
    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 503
    assert "Test model missing error" in resp_ready.json()["detail"]

    # Post analyze should gracefully return 503 with error alert
    resp_analyze = client.post(
        "/analyze",
        files={"file": ("test_xray.png", valid_png_bytes, "image/png")}
    )
    assert resp_analyze.status_code == 503
    assert "Test model missing error" in resp_analyze.text


# Case 8: Classification failure is logged and raises diagnosable error
def test_08_classification_failure_logged(monkeypatch):
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    def fail_classification(cls, img):
        raise ValueError("Simulated classification failure")
    monkeypatch.setattr(app_module, "predict_classification", fail_classification)

    with pytest.raises(RuntimeError) as exc_info:
        analyze_single_xray(dummy_img)
    assert "Classification failed" in str(exc_info.value)


# Case 9: Segmentation failure is logged and raises diagnosable error
def test_09_segmentation_failure_logged(monkeypatch):
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    def mock_cls(cls, img):
        return {"predicted_class": "Normal", "confidence": 0.95, "pred_index": 2, "class_probabilities": {}}
    def fail_seg(seg, img):
        raise ValueError("Simulated segmentation failure")

    monkeypatch.setattr(app_module, "predict_classification", mock_cls)
    monkeypatch.setattr(app_module, "predict_segmentation", fail_seg)

    with pytest.raises(RuntimeError) as exc_info:
        analyze_single_xray(dummy_img)
    assert "Segmentation failed" in str(exc_info.value)


# Case 10: Grad-CAM failure is logged and raises diagnosable error
def test_10_gradcam_failure_logged(monkeypatch):
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    def mock_cls(cls, img):
        return {"predicted_class": "Normal", "confidence": 0.95, "pred_index": 2, "class_probabilities": {}}
    def mock_seg(seg, img):
        return {"mask_area_pct": 25.0, "mask_binary": dummy_img[..., 0], "overlay": dummy_img}
    def fail_gcam(cls, img, target_pred_index):
        raise ValueError("Simulated Grad-CAM failure")

    monkeypatch.setattr(app_module, "predict_classification", mock_cls)
    monkeypatch.setattr(app_module, "predict_segmentation", mock_seg)
    monkeypatch.setattr(app_module, "generate_gradcam", fail_gcam)

    with pytest.raises(RuntimeError) as exc_info:
        analyze_single_xray(dummy_img)
    assert "Grad-CAM generation failed" in str(exc_info.value)


# Case 11: Repeated analyses do not accumulate excessive memory
def test_11_repeated_analyses_memory_stability(valid_png_bytes):
    process = psutil.Process(os.getpid())
    gc.collect()
    initial_rss = process.memory_info().rss / (1024 * 1024)

    for i in range(4):
        resp = client.post("/analyze", data={"sample": "sample_1"})
        assert resp.status_code == 200

    gc.collect()
    final_rss = process.memory_info().rss / (1024 * 1024)
    growth = final_rss - initial_rss
    # Growth across 4 iterations should be modest (< 60 MB) and not continuously accumulating
    assert growth < 60, f"Memory growth of {growth:.2f} MB exceeds stability threshold"


# Case 12: Comparison of two valid images works
def test_12_comparison_two_valid_images():
    response = client.post(
        "/compare",
        data={"run_comparison": "true", "use_demo1": "true", "use_demo2": "true"}
    )
    assert response.status_code == 200
    assert "Model Output Comparison" in response.text
    assert "Measurable Model-Output Differences" in response.text or "Quantitative Output Differences" in response.text
    assert "Per-Class Probability Shift" in response.text


# Case 13: Existing report export functionality remains operational
def test_13_report_export_operational():
    # Run an analysis first to ensure latest report is cached
    resp_an = client.post("/analyze", data={"sample": "sample_1"})
    assert resp_an.status_code == 200

    # Test TXT export
    resp_txt = client.get("/export/txt")
    assert resp_txt.status_code == 200
    assert "text/plain" in resp_txt.headers["content-type"]
    assert "CHESTVISION AI - MODEL ANALYSIS REPORT" in resp_txt.text

    # Test JSON export
    resp_json = client.get("/export/json")
    assert resp_json.status_code == 200
    assert "application/json" in resp_json.headers["content-type"]
    data = resp_json.json()
    assert "classification" in data
    assert "segmentation" in data
    assert "explainability" in data
