import os
import io
import time
import base64
from PIL import Image
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    CLASS_NAMES, CLASS_COLORS, MEDICAL_DISCLAIMER, COMPARISON_DISCLAIMER,
    CLASSIFIER_PATH, SEGMENTER_PATH, IMG_SIZE
)
from utils.model_loader import load_models
from utils.preprocessing import load_and_preprocess_pil_image
from utils.classification import predict_classification
from utils.segmentation import predict_segmentation
from utils.gradcam import generate_gradcam
from utils.comparison import compare_xrays
from utils.report import generate_text_report, generate_json_report

# Helper function: Convert NumPy RGB uint8 array to Base64 PNG data URL
def image_to_base64(img_uint8):
    if img_uint8 is None:
        return ""
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

# Initialize FastAPI App
app = FastAPI(
    title="ChestVision AI",
    description="Explainable Chest X-ray Analysis Dashboard powered by FastAPI & TensorFlow",
    version="1.0"
)

# Mount Static Assets
base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, "assets")
templates_dir = os.path.join(base_dir, "templates")

app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# Initialize Jinja2 Templates
templates = Jinja2Templates(directory=templates_dir)

# Load Deep Learning Models
try:
    classifier, segmenter = load_models()
    models_loaded = True
    model_error_msg = ""
except Exception as e:
    classifier, segmenter = None, None
    models_loaded = False
    model_error_msg = str(e)

# In-memory storage for export reports
cache_report = {}

def analyze_single_xray(raw_img_uint8):
    cls_res = predict_classification(classifier, raw_img_uint8)
    seg_res = predict_segmentation(segmenter, raw_img_uint8)
    gcam_res = generate_gradcam(classifier, raw_img_uint8, target_pred_index=cls_res["pred_index"])

    analysis = {
        "raw_image": raw_img_uint8,
        "classification": cls_res,
        "segmentation": seg_res,
        "gradcam": gcam_res,
        "timestamp": time.strftime("%H:%M:%S")
    }
    return analysis


# ==============================================================================
# FASTAPI ROUTES
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
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )


@app.post("/analyze", response_class=HTMLResponse)
async def route_analyze_post(
    request: Request,
    file: UploadFile = File(None),
    sample: str = Form(None)
):
    if not models_loaded:
        raise HTTPException(status_code=500, detail=f"Models not loaded: {model_error_msg}")

    raw_img_uint8 = None
    sample_dir = os.path.join(base_dir, "assets", "samples")

    if sample == "sample_1":
        s_path = os.path.join(sample_dir, "sample_xray_1.png")
        if os.path.exists(s_path):
            raw_img_uint8 = load_and_preprocess_pil_image(s_path)
    elif sample == "sample_2":
        s_path = os.path.join(sample_dir, "sample_xray_2.png")
        if os.path.exists(s_path):
            raw_img_uint8 = load_and_preprocess_pil_image(s_path)
    elif file is not None and file.filename != "":
        contents = await file.read()
        raw_img_uint8 = load_and_preprocess_pil_image(io.BytesIO(contents))

    results = None
    if raw_img_uint8 is not None:
        analysis_res = analyze_single_xray(raw_img_uint8)
        cache_report["latest"] = analysis_res

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

    return templates.TemplateResponse(
        request=request,
        name="analyze.html",
        context={
            "active_page": "analyze",
            "models_loaded": models_loaded,
            "model_error": model_error_msg,
            "results": results,
            "class_colors": CLASS_COLORS,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }
    )


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
    if not models_loaded:
        raise HTTPException(status_code=500, detail=f"Models not loaded: {model_error_msg}")

    sample_dir = os.path.join(base_dir, "assets", "samples")
    img1_np, img2_np = None, None

    # Load Image 1
    if use_demo1:
        s_path = os.path.join(sample_dir, "sample_xray_1.png")
        if os.path.exists(s_path):
            img1_np = load_and_preprocess_pil_image(s_path)
    elif file1 is not None and file1.filename != "":
        contents1 = await file1.read()
        img1_np = load_and_preprocess_pil_image(io.BytesIO(contents1))

    # Load Image 2
    if use_demo2:
        s_path = os.path.join(sample_dir, "sample_xray_2.png")
        if os.path.exists(s_path):
            img2_np = load_and_preprocess_pil_image(s_path)
    elif file2 is not None and file2.filename != "":
        contents2 = await file2.read()
        img2_np = load_and_preprocess_pil_image(io.BytesIO(contents2))

    # Default to sample 1 and sample 2 if both are empty when run_comparison requested
    if img1_np is None:
        s_path1 = os.path.join(sample_dir, "sample_xray_1.png")
        if os.path.exists(s_path1):
            img1_np = load_and_preprocess_pil_image(s_path1)

    if img2_np is None:
        s_path2 = os.path.join(sample_dir, "sample_xray_2.png")
        if os.path.exists(s_path2):
            img2_np = load_and_preprocess_pil_image(s_path2)

    res1 = analyze_single_xray(img1_np)
    res2 = analyze_single_xray(img2_np)
    cmp_res = compare_xrays(res1, res2)

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
        raise HTTPException(status_code=404, detail="No recent analysis available to export.")

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
        raise HTTPException(status_code=400, detail="Invalid export format. Choose 'txt' or 'json'.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )
