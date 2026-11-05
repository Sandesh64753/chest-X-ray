import os
import io
import time
import base64
import logging
import gc
import asyncio
from PIL import Image, UnidentifiedImageError
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from config import (
    CLASS_NAMES, CLASS_COLORS, MEDICAL_DISCLAIMER, COMPARISON_DISCLAIMER,
    CLASSIFIER_PATH, SEGMENTER_PATH, IMG_SIZE,
    MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_CONTENT_TYPES
)
from utils.model_loader import load_models
from utils.preprocessing import load_and_preprocess_pil_image
from utils.classification import predict_classification
from utils.segmentation import predict_segmentation
from utils.gradcam import generate_gradcam
from utils.comparison import compare_xrays
from utils.report import generate_text_report, generate_json_report

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("chestvision")

# Initialize Concurrency Lock for Heavy ML Inference
inference_lock = asyncio.Lock()


def image_to_base64(img_uint8: np.ndarray) -> str:
    """Convert NumPy RGB uint8 array to Base64 PNG data URL."""
    if img_uint8 is None:
        return ""
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


# Initialize FastAPI App
app = FastAPI(
    title="ChestVision AI API",
    description="REST API for Explainable Chest X-ray Analysis powered by FastAPI & TensorFlow",
    version="2.0"
)

# CORS — allow React frontend (dev + prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sample images for static access
base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, "assets")
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# Load Deep Learning Models at Startup
try:
    logger.info("Initializing deep learning models from disk...")
    t_load_start = time.perf_counter()
    classifier, segmenter = load_models()
    models_loaded = True
    model_error_msg = ""
    logger.info("Models loaded successfully in %.3fs.", time.perf_counter() - t_load_start)
except Exception as e:
    classifier, segmenter = None, None
    models_loaded = False
    model_error_msg = str(e)
    logger.exception("Critical error loading deep learning models: %s", str(e))

# In-memory cache for latest analysis (export)
cache_report = {}


def analyze_single_xray(raw_img_uint8: np.ndarray) -> dict:
    """
    Synchronous analysis pipeline: classification, segmentation, Grad-CAM.
    Runs inside run_in_threadpool so asyncio event loop is never blocked.
    """
    t_cls_start = time.perf_counter()
    logger.info("Stage 4/8: Classification inference started...")
    try:
        cls_res = predict_classification(classifier, raw_img_uint8)
        logger.info(
            "Stage 4/8: Classification completed in %.3fs -> %s (%.2f%% confidence)",
            time.perf_counter() - t_cls_start,
            cls_res["predicted_class"],
            cls_res["confidence"] * 100.0
        )
    except Exception as exc:
        logger.exception("Classification failure: %s", str(exc))
        raise RuntimeError(f"Classification failed: {str(exc)}") from exc

    t_seg_start = time.perf_counter()
    logger.info("Stage 5/8: Segmentation inference started...")
    try:
        seg_res = predict_segmentation(segmenter, raw_img_uint8)
        logger.info(
            "Stage 5/8: Segmentation completed in %.3fs -> Mask area: %.2f%%",
            time.perf_counter() - t_seg_start,
            seg_res["mask_area_pct"]
        )
    except Exception as exc:
        logger.exception("Segmentation failure: %s", str(exc))
        raise RuntimeError(f"Segmentation failed: {str(exc)}") from exc

    t_gcam_start = time.perf_counter()
    logger.info("Stage 6/8: Grad-CAM generation started...")
    try:
        gcam_res = generate_gradcam(classifier, raw_img_uint8, target_pred_index=cls_res["pred_index"])
        logger.info(
            "Stage 6/8: Grad-CAM completed in %.3fs -> Region: %s",
            time.perf_counter() - t_gcam_start,
            gcam_res["region_description"]
        )
    except Exception as exc:
        logger.exception("Grad-CAM generation failure: %s", str(exc))
        raise RuntimeError(f"Grad-CAM generation failed: {str(exc)}") from exc

    gc.collect()

    return {
        "raw_image": raw_img_uint8,
        "classification": cls_res,
        "segmentation": seg_res,
        "gradcam": gcam_res,
        "timestamp": time.strftime("%H:%M:%S")
    }


async def read_and_validate_upload(file: UploadFile) -> np.ndarray:
    """Validates + decodes an uploaded image into a 224x224x3 uint8 array."""
    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file selected for upload."
        )

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        logger.warning("Unusual content-type '%s' for '%s'", file.content_type, file.filename)

    bytes_buffer = bytearray()
    chunk_size = 64 * 1024
    total_read = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB."
            )
        bytes_buffer.extend(chunk)

    if total_read == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    logger.info(
        "Stage 2/8: Upload validated - filename='%s', size=%.2f KB",
        file.filename, total_read / 1024.0
    )

    try:
        raw_img_uint8 = load_and_preprocess_pil_image(io.BytesIO(bytes_buffer))
        return raw_img_uint8
    except UnidentifiedImageError as uie:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or supported image."
        ) from uie
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {str(exc)}"
        ) from exc


# ==============================================================================
# HEALTH ENDPOINTS
# ==============================================================================

@app.get("/api/health")
async def route_health():
    """Lightweight health check for Docker and load balancer."""
    return {
        "status": "healthy" if models_loaded else "degraded",
        "service": "ChestVision AI",
        "models_loaded": models_loaded,
        "error": model_error_msg if not models_loaded else None
    }


@app.get("/api/ready")
async def route_ready():
    """Readiness check verifying models are loaded."""
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Models not ready: {model_error_msg}"
        )
    return {
        "status": "ready",
        "classifier_loaded": classifier is not None,
        "segmenter_loaded": segmenter is not None
    }


# ==============================================================================
# ANALYSIS ENDPOINTS
# ==============================================================================

@app.post("/api/analyze")
async def route_analyze(
    file: UploadFile = File(None),
    sample: str = Form(None)
):
    """
    Accepts an uploaded chest X-ray or a sample key ('sample_1', 'sample_2').
    Returns JSON with classification, segmentation, and Grad-CAM results.
    """
    t_req_start = time.perf_counter()
    logger.info("Stage 1/8: POST /api/analyze received")

    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Deep learning models are currently unavailable."
        )

    raw_img_uint8 = None
    sample_dir = os.path.join(base_dir, "assets", "samples")

    if sample == "sample_1":
        s_path = os.path.join(sample_dir, "sample_xray_1.png")
        if os.path.exists(s_path):
            logger.info("Stage 2/8: Selected Demo Sample 1")
            raw_img_uint8 = load_and_preprocess_pil_image(s_path)
        else:
            raise HTTPException(status_code=404, detail="Demo Sample 1 not found on server.")
    elif sample == "sample_2":
        s_path = os.path.join(sample_dir, "sample_xray_2.png")
        if os.path.exists(s_path):
            logger.info("Stage 2/8: Selected Demo Sample 2")
            raw_img_uint8 = load_and_preprocess_pil_image(s_path)
        else:
            raise HTTPException(status_code=404, detail="Demo Sample 2 not found on server.")
    elif file is not None and file.filename:
        raw_img_uint8 = await read_and_validate_upload(file)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a chest X-ray image or choose a demo sample."
        )

    try:
        async with inference_lock:
            analysis_res = await run_in_threadpool(analyze_single_xray, raw_img_uint8)

        cache_report["latest"] = analysis_res

        pred_cls = analysis_res["classification"]["predicted_class"]
        pred_color = CLASS_COLORS.get(pred_cls, "#0284c7")

        # Serialize numpy arrays -> base64 for JSON transport
        result = {
            "classification": {
                "predicted_class": analysis_res["classification"]["predicted_class"],
                "confidence": analysis_res["classification"]["confidence"],
                "class_probabilities": analysis_res["classification"]["class_probabilities"],
                "pred_color": pred_color,
            },
            "segmentation": {
                "mask_area_pct": analysis_res["segmentation"]["mask_area_pct"],
            },
            "gradcam": {
                "region_description": analysis_res["gradcam"]["region_description"],
            },
            "images": {
                "raw": image_to_base64(raw_img_uint8),
                "mask": image_to_base64(analysis_res["segmentation"]["mask_binary"]),
                "seg_overlay": image_to_base64(analysis_res["segmentation"]["overlay"]),
                "heatmap": image_to_base64(analysis_res["gradcam"]["heatmap_color"]),
                "gcam_overlay": image_to_base64(analysis_res["gradcam"]["overlay"]),
            },
            "timestamp": analysis_res["timestamp"],
            "class_colors": CLASS_COLORS,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }

        logger.info(
            "Stage 8/8: Analysis complete in %.3fs total.",
            time.perf_counter() - t_req_start
        )
        return JSONResponse(content=result)

    except Exception as exc:
        logger.exception("Inference execution failed: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during X-ray analysis. Please try again."
        ) from exc


@app.post("/api/compare")
async def route_compare(
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    use_demo1: str = Form(None),
    use_demo2: str = Form(None),
):
    """
    Accepts two X-rays (uploaded or demo), runs analysis on each, and returns
    quantitative comparison metrics as JSON.
    """
    logger.info("POST /api/compare received")

    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Deep learning models are currently unavailable."
        )

    sample_dir = os.path.join(base_dir, "assets", "samples")
    img1_np, img2_np = None, None

    # --- Load Image 1 ---
    if use_demo1:
        s_path = os.path.join(sample_dir, "sample_xray_1.png")
        if os.path.exists(s_path):
            img1_np = load_and_preprocess_pil_image(s_path)
        else:
            raise HTTPException(status_code=404, detail="Demo Sample 1 not found on server.")
    elif file1 is not None and file1.filename:
        img1_np = await read_and_validate_upload(file1)

    # --- Load Image 2 ---
    if use_demo2:
        s_path = os.path.join(sample_dir, "sample_xray_2.png")
        if os.path.exists(s_path):
            img2_np = load_and_preprocess_pil_image(s_path)
        else:
            raise HTTPException(status_code=404, detail="Demo Sample 2 not found on server.")
    elif file2 is not None and file2.filename:
        img2_np = await read_and_validate_upload(file2)

    if img1_np is None or img2_np is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide or select both Image 1 and Image 2 to run comparison."
        )

    try:
        async with inference_lock:
            res1 = await run_in_threadpool(analyze_single_xray, img1_np)
            res2 = await run_in_threadpool(analyze_single_xray, img2_np)
            cmp_res = await run_in_threadpool(compare_xrays, res1, res2)

        diffs = cmp_res["measurable_differences"]
        delta_table = []
        for cls in CLASS_NAMES:
            p1 = cmp_res["image_1"]["probabilities"][cls] * 100
            p2 = cmp_res["image_2"]["probabilities"][cls] * 100
            d = diffs["class_probability_deltas"][cls] * 100
            delta_table.append({"cls": cls, "p1": round(p1, 2), "p2": round(p2, 2), "d": round(d, 2)})

        result = {
            "image_1": {
                **cmp_res["image_1"],
                "images": {
                    "raw": image_to_base64(img1_np),
                    "gcam_overlay": image_to_base64(res1["gradcam"]["overlay"]),
                    "mask": image_to_base64(res1["segmentation"]["mask_binary"]),
                    "seg_overlay": image_to_base64(res1["segmentation"]["overlay"]),
                }
            },
            "image_2": {
                **cmp_res["image_2"],
                "images": {
                    "raw": image_to_base64(img2_np),
                    "gcam_overlay": image_to_base64(res2["gradcam"]["overlay"]),
                    "mask": image_to_base64(res2["segmentation"]["mask_binary"]),
                    "seg_overlay": image_to_base64(res2["segmentation"]["overlay"]),
                }
            },
            "measurable_differences": diffs,
            "delta_table": delta_table,
            "disclaimer": COMPARISON_DISCLAIMER,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "class_colors": CLASS_COLORS,
        }
        return JSONResponse(content=result)

    except Exception as exc:
        logger.exception("Comparison analysis failure: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during comparative analysis. Please try again."
        ) from exc


@app.get("/api/export/{fmt}")
async def route_export_report(fmt: str):
    """Export latest analysis as 'txt' or 'json'."""
    if "latest" not in cache_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recent analysis available to export."
        )

    analysis_res = cache_report["latest"]
    ts = analysis_res["timestamp"].replace(":", "")

    if fmt == "txt":
        content = generate_text_report(analysis_res)
        filename = f"chestvision_report_{ts}.txt"
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    elif fmt == "json":
        content = generate_json_report(analysis_res)
        filename = f"chestvision_data_{ts}.json"
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid export format. Choose 'txt' or 'json'."
        )


@app.get("/api/info")
async def route_info():
    """Returns static info about the model architecture and class definitions."""
    return {
        "app_name": "ChestVision AI",
        "version": "2.0",
        "models": {
            "classifier": {
                "architecture": "DenseNet121",
                "task": "Multi-class chest X-ray classification",
                "classes": CLASS_NAMES,
                "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
                "training": "Two-stage fine-tuning on COVID-19 Radiography Dataset",
            },
            "segmenter": {
                "architecture": "U-Net",
                "task": "Lung field segmentation",
                "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
            },
        },
        "class_colors": CLASS_COLORS,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
        "models_loaded": models_loaded,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting Uvicorn server on 0.0.0.0:%d", port)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=75
    )
