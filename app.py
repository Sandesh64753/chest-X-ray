import os
import io
import time
import base64
import logging
import gc
import asyncio
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
# Limits simultaneous execution to 1 to prevent memory exhaustion in container environments
inference_lock = asyncio.Lock()

# Helper function: Convert NumPy RGB uint8 array to Base64 PNG data URL
def image_to_base64(img_uint8: np.ndarray) -> str:
    if img_uint8 is None:
        return ""
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

# Initialize FastAPI App
app = FastAPI(
    title="ChestVision AI",
    description="Explainable Chest X-ray Analysis Dashboard powered by FastAPI & TensorFlow",
    version="1.1"
)

# Mount Static Assets & Templates
base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, "assets")
templates_dir = os.path.join(base_dir, "templates")

app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
templates = Jinja2Templates(directory=templates_dir)

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

# In-memory storage for export reports
cache_report = {}


def analyze_single_xray(raw_img_uint8: np.ndarray) -> dict:
    """
    Synchronous analysis pipeline executing classification, segmentation, and Grad-CAM.
    Designed to run inside run_in_threadpool so the asyncio event loop is never blocked.
    """
    # Stage 4: Classification
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

    # Stage 5: Segmentation
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

    # Stage 6: Grad-CAM Explainability
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

    # Force memory garbage collection after tensor operations
    gc.collect()

    analysis = {
        "raw_image": raw_img_uint8,
        "classification": cls_res,
        "segmentation": seg_res,
        "gradcam": gcam_res,
        "timestamp": time.strftime("%H:%M:%S")
    }
    return analysis


async def read_and_validate_upload(file: UploadFile) -> np.ndarray:
    """
    Safely reads and validates uploaded file with chunked bounds checking
    and decodes into a verified 224x224x3 uint8 array.
    """
    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file selected for upload."
        )

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    # Validate content-type if provided
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        logger.warning("Unusual content-type '%s' for file '%s', checking extension.", file.content_type, file.filename)

    # Chunked read up to MAX_UPLOAD_SIZE_BYTES to prevent memory saturation
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
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB."
            )
        bytes_buffer.extend(chunk)

    if total_read == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    logger.info(
        "Stage 2/8: Upload validated - filename='%s', content_type='%s', size=%.2f KB",
        file.filename, file.content_type, total_read / 1024.0
    )

    # Stage 3: Image decoding and preprocessing
    t_decode_start = time.perf_counter()
    try:
        raw_img_uint8 = load_and_preprocess_pil_image(io.BytesIO(bytes_buffer))
        logger.info("Stage 3/8: Image decoded and resized to (%d, %d, 3) in %.3fs", IMG_SIZE, IMG_SIZE, time.perf_counter() - t_decode_start)
        return raw_img_uint8
    except UnidentifiedImageError as uie:
        logger.warning("Uploaded file is not a valid decodable image: %s", str(uie))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or supported image."
        ) from uie
    except Exception as exc:
        logger.exception("Unexpected error while decoding image: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {str(exc)}"
        ) from exc


# ==============================================================================
# HEALTH & READINESS ENDPOINTS
# ==============================================================================

@app.get("/health")
async def route_health():
    """
    Lightweight healthcheck endpoint for Docker and Render load balancer.
    Does not execute expensive model operations.
    """
    return {
        "status": "healthy" if models_loaded else "degraded",
        "service": "ChestVision AI",
        "models_loaded": models_loaded,
        "error": model_error_msg if not models_loaded else None
    }


@app.get("/ready")
async def route_ready():
    """
    Readiness endpoint verifying that models are present and ready for inference.
    """
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
# FASTAPI APPLICATION ROUTES
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def route_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "active_page": "dashboard",
            "models_loaded": models_loaded,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )


@app.get("/analyze", response_class=HTMLResponse)
async def route_analyze_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="analyze.html",
        context={
            "active_page": "analyze",
            "models_loaded": models_loaded,
            "model_error": model_error_msg,
            "results": None,
            "upload_error": None,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )


@app.post("/analyze", response_class=HTMLResponse)
async def route_analyze_post(
    request: Request,
    file: UploadFile = File(None),
    sample: str = Form(None)
):
    t_req_start = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    logger.info("Stage 1/8: Request received - POST /analyze from client %s", client_host)

    if not models_loaded:
        logger.error("Analysis requested while models are not loaded: %s", model_error_msg)
        return templates.TemplateResponse(
            request=request,
            name="analyze.html",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            context={
                "active_page": "analyze",
                "models_loaded": False,
                "model_error": model_error_msg,
                "results": None,
                "upload_error": "Deep learning models are currently unavailable.",
                "medical_disclaimer": MEDICAL_DISCLAIMER
            }
        )

    raw_img_uint8 = None
    upload_error = None
    sample_dir = os.path.join(base_dir, "assets", "samples")

    try:
        # Check sample selection
        if sample == "sample_1":
            s_path = os.path.join(sample_dir, "sample_xray_1.png")
            if os.path.exists(s_path):
                logger.info("Stage 2/8: Selected Demo Sample 1: %s", s_path)
                raw_img_uint8 = load_and_preprocess_pil_image(s_path)
            else:
                upload_error = "Demo Sample 1 file not found on server."
        elif sample == "sample_2":
            s_path = os.path.join(sample_dir, "sample_xray_2.png")
            if os.path.exists(s_path):
                logger.info("Stage 2/8: Selected Demo Sample 2: %s", s_path)
                raw_img_uint8 = load_and_preprocess_pil_image(s_path)
            else:
                upload_error = "Demo Sample 2 file not found on server."
        elif file is not None and file.filename:
            raw_img_uint8 = await read_and_validate_upload(file)
        else:
            upload_error = "Please upload a chest X-ray image (PNG, JPG, JPEG) or choose a demo sample."

    except HTTPException as he:
        upload_error = he.detail
        logger.warning("Upload validation rejected with status %d: %s", he.status_code, he.detail)
    except Exception as exc:
        upload_error = f"Error processing uploaded image: {str(exc)}"
        logger.exception("Unexpected error in upload handling: %s", str(exc))

    if upload_error or raw_img_uint8 is None:
        return templates.TemplateResponse(
            request=request,
            name="analyze.html",
            status_code=status.HTTP_400_BAD_REQUEST,
            context={
                "active_page": "analyze",
                "models_loaded": models_loaded,
                "model_error": model_error_msg,
                "results": None,
                "upload_error": upload_error or "No image available for analysis.",
                "class_colors": CLASS_COLORS,
                "medical_disclaimer": MEDICAL_DISCLAIMER
            }
        )

    # Execute ML Inference under concurrency lock using threadpool
    results = None
    try:
        async with inference_lock:
            analysis_res = await run_in_threadpool(analyze_single_xray, raw_img_uint8)

        cache_report["latest"] = analysis_res

        # Stage 7: Base64 encoding
        t_b64_start = time.perf_counter()
        pred_cls = analysis_res["classification"]["predicted_class"]
        pred_color = CLASS_COLORS.get(pred_cls, "#0284c7")

        results = {
            "classification": analysis_res["classification"],
            "segmentation": analysis_res["segmentation"],
            "gradcam": analysis_res["gradcam"],
            "pred_color": pred_color,
            "raw_img_b64": image_to_base64(raw_img_uint8),
            "mask_b64": image_to_base64(analysis_res["segmentation"]["mask_binary"]),
            "seg_overlay_b64": image_to_base64(analysis_res["segmentation"]["overlay"]),
            "heatmap_b64": image_to_base64(analysis_res["gradcam"]["heatmap_color"]),
            "gcam_overlay_b64": image_to_base64(analysis_res["gradcam"]["overlay"]),
            "timestamp": analysis_res["timestamp"]
        }
        logger.info("Stage 7/8: Base64 image encoding completed in %.3fs", time.perf_counter() - t_b64_start)

    except Exception as exc:
        logger.exception("Inference execution failed: %s", str(exc))
        return templates.TemplateResponse(
            request=request,
            name="analyze.html",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            context={
                "active_page": "analyze",
                "models_loaded": models_loaded,
                "model_error": model_error_msg,
                "results": None,
                "upload_error": "An internal error occurred during X-ray analysis. Please try again.",
                "class_colors": CLASS_COLORS,
                "medical_disclaimer": MEDICAL_DISCLAIMER
            }
        )

    # Stage 8: Template rendering & response dispatch
    t_render_start = time.perf_counter()
    response = templates.TemplateResponse(
        request=request,
        name="analyze.html",
        context={
            "active_page": "analyze",
            "models_loaded": models_loaded,
            "model_error": model_error_msg,
            "results": results,
            "upload_error": None,
            "class_colors": CLASS_COLORS,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )
    t_total = time.perf_counter() - t_req_start
    logger.info(
        "Stage 8/8: Template rendered in %.3fs. Request completed in %.3fs total.",
        time.perf_counter() - t_render_start, t_total
    )
    return response


@app.get("/compare", response_class=HTMLResponse)
async def route_compare_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "active_page": "compare",
            "models_loaded": models_loaded,
            "model_error": model_error_msg,
            "comparison": None,
            "upload_error": None,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )


@app.post("/compare", response_class=HTMLResponse)
async def route_compare_post(
    request: Request,
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    use_demo1: str = Form(None),
    use_demo2: str = Form(None),
    run_comparison: str = Form(None)
):
    logger.info("Request received - POST /compare")

    if not models_loaded:
        return templates.TemplateResponse(
            request=request,
            name="compare.html",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            context={
                "active_page": "compare",
                "models_loaded": False,
                "model_error": model_error_msg,
                "comparison": None,
                "upload_error": "Deep learning models are currently unavailable.",
                "medical_disclaimer": MEDICAL_DISCLAIMER
            }
        )

    sample_dir = os.path.join(base_dir, "assets", "samples")
    img1_np, img2_np = None, None
    upload_error = None

    try:
        # Load Image 1
        if use_demo1:
            s_path = os.path.join(sample_dir, "sample_xray_1.png")
            if os.path.exists(s_path):
                img1_np = load_and_preprocess_pil_image(s_path)
        elif file1 is not None and file1.filename:
            img1_np = await read_and_validate_upload(file1)

        # Load Image 2
        if use_demo2:
            s_path = os.path.join(sample_dir, "sample_xray_2.png")
            if os.path.exists(s_path):
                img2_np = load_and_preprocess_pil_image(s_path)
        elif file2 is not None and file2.filename:
            img2_np = await read_and_validate_upload(file2)

        # Fallback to demo images if run_comparison requested without explicit images
        if run_comparison:
            if img1_np is None:
                s_path1 = os.path.join(sample_dir, "sample_xray_1.png")
                if os.path.exists(s_path1):
                    img1_np = load_and_preprocess_pil_image(s_path1)
            if img2_np is None:
                s_path2 = os.path.join(sample_dir, "sample_xray_2.png")
                if os.path.exists(s_path2):
                    img2_np = load_and_preprocess_pil_image(s_path2)

        if img1_np is None or img2_np is None:
            upload_error = "Please provide or select both Image 1 and Image 2 to run comparison."

    except HTTPException as he:
        upload_error = he.detail
    except Exception as exc:
        upload_error = f"Error processing images for comparison: {str(exc)}"
        logger.exception("Error in comparison upload handling: %s", str(exc))

    if upload_error or img1_np is None or img2_np is None:
        return templates.TemplateResponse(
            request=request,
            name="compare.html",
            status_code=status.HTTP_400_BAD_REQUEST,
            context={
                "active_page": "compare",
                "models_loaded": models_loaded,
                "model_error": model_error_msg,
                "comparison": None,
                "upload_error": upload_error,
                "medical_disclaimer": MEDICAL_DISCLAIMER
            }
        )

    # Execute both analyses under lock via threadpool
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
            delta_table.append({"cls": cls, "p1": p1, "p2": p2, "d": d})

        return templates.TemplateResponse(
            request=request,
            name="compare.html",
            context={
                "active_page": "compare",
                "models_loaded": models_loaded,
                "model_error": model_error_msg,
                "img1_b64": image_to_base64(img1_np),
                "img2_b64": image_to_base64(img2_np),
                "res1": res1,
                "res2": res2,
                "comparison": cmp_res,
                "diffs": diffs,
                "delta_table": delta_table,
                "res1_gcam_overlay_b64": image_to_base64(res1["gradcam"]["overlay"]),
                "res1_mask_b64": image_to_base64(res1["segmentation"]["mask_binary"]),
                "res2_gcam_overlay_b64": image_to_base64(res2["gradcam"]["overlay"]),
                "res2_mask_b64": image_to_base64(res2["segmentation"]["mask_binary"]),
                "upload_error": None,
                "medical_disclaimer": MEDICAL_DISCLAIMER
            }
        )
    except Exception as exc:
        logger.exception("Comparison analysis failure: %s", str(exc))
        return templates.TemplateResponse(
            request=request,
            name="compare.html",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            context={
                "active_page": "compare",
                "models_loaded": models_loaded,
                "model_error": model_error_msg,
                "comparison": None,
                "upload_error": "An internal error occurred during comparative analysis. Please try again.",
                "medical_disclaimer": MEDICAL_DISCLAIMER
            }
        )


@app.get("/explainability", response_class=HTMLResponse)
async def route_explainability(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="explainability.html",
        context={
            "active_page": "explainability",
            "models_loaded": models_loaded,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )


@app.get("/model-info", response_class=HTMLResponse)
async def route_model_info(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="model_info.html",
        context={
            "active_page": "model_info",
            "models_loaded": models_loaded,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )


@app.get("/about", response_class=HTMLResponse)
async def route_about(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "active_page": "about",
            "models_loaded": models_loaded,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )


@app.get("/export/{fmt}")
async def route_export_report(fmt: str):
    if "latest" not in cache_report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No recent analysis available to export.")

    analysis_res = cache_report["latest"]
    if fmt == "txt":
        content = generate_text_report(analysis_res)
        filename = f"chestvision_report_{analysis_res['timestamp'].replace(':','')}.txt"
        return Response(content=content, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename={filename}"})
    elif fmt == "json":
        content = generate_json_report(analysis_res)
        filename = f"chestvision_data_{analysis_res['timestamp'].replace(':','')}.json"
        return Response(content=content, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename}"})
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid export format. Choose 'txt' or 'json'.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    logger.info("Starting Uvicorn server on 0.0.0.0:%d", port)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=75
    )
