"""FastAPI Backend for Citizen-to-Municipal Road Damage Grievance Portal.

Isolated 100% inside app/. Runs inference on CPU and interacts with
municipal databases and Gmail SMTP.
"""

import base64
import os
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .detector import get_detector
from .mailer import is_smtp_configured, send_grievance_email
from .router import (
    generate_complaint_draft,
    load_municipalities_catalog,
    match_municipality,
    reverse_geocode,
)

app = FastAPI(
    title="Citizen-to-Municipal Road Damage Grievance Portal",
    description="Automated pavement distress AI auditor and municipal routing system.",
    version="1.0.0",
)

# Enable CORS for local testing and mobile webview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SendReportRequest(BaseModel):
    to_email: str
    subject: str
    body: str
    image_base64: Optional[str] = None
    cc_citizen_email: Optional[str] = None
    citizen_name: Optional[str] = None


class GeocodeRequest(BaseModel):
    lat: float
    lon: float
    location_text: Optional[str] = None


@app.get("/")
def serve_index():
    """Serves mobile-friendly single page web frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Road Damage Reporting App Backend Running. Static index.html not found."}


@app.get("/api/status")
def get_system_status():
    """Returns runtime health and configuration."""
    detector = get_detector()
    catalog = load_municipalities_catalog()
    return {
        "status": "online",
        "device": "cpu",
        "active_weights": os.path.basename(detector.weights_path),
        "weights_path": detector.weights_path,
        "smtp_configured": is_smtp_configured(),
        "seeded_municipalities_count": len(catalog.get("municipalities", [])),
    }


@app.post("/api/geocode")
def geocode_endpoint(req: GeocodeRequest):
    """Performs reverse geocoding and municipality matching."""
    geo_data = reverse_geocode(req.lat, req.lon)
    muni_info = match_municipality(location_text=req.location_text or "", geo_data=geo_data)
    return {
        "geo": geo_data,
        "municipality": muni_info,
    }


@app.post("/api/analyze")
async def analyze_road_photo(
    image: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    city: Optional[str] = Form(None),
    landmark: Optional[str] = Form(None),
    citizen_name: Optional[str] = Form(None),
    citizen_phone: Optional[str] = Form(None),
    citizen_email: Optional[str] = Form(None),
    conf_threshold: Optional[float] = Form(0.15),
):
    """Audits uploaded road photograph for structural distress, matches municipality, and drafts complaint."""
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        detector = get_detector()
        detection_result = detector.detect_and_annotate(
            image_bytes=image_bytes,
            conf_threshold=conf_threshold if conf_threshold is not None else 0.15,
        )

        # Reverse geocoding & Municipal routing
        geo_data = None
        if lat is not None and lon is not None:
            geo_data = reverse_geocode(lat, lon)

        muni_info = match_municipality(
            location_text=city or landmark or "",
            geo_data=geo_data,
        )

        address_display = ""
        if geo_data and geo_data.get("display_name"):
            address_display = geo_data["display_name"]
        elif city:
            address_display = city

        # Generate official grievance letter
        draft = generate_complaint_draft(
            citizen_name=citizen_name or "",
            citizen_phone=citizen_phone or "",
            citizen_email=citizen_email or "",
            landmark=landmark or "",
            lat=lat,
            lon=lon,
            address_display=address_display,
            detection_summary=detection_result,
            municipality_info=muni_info,
        )

        return {
            "success": True,
            "detection": {
                "model_version": detection_result["model_version"],
                "annotated_image_data_url": detection_result["annotated_image_data_url"],
                "total_damage_count": detection_result["total_damage_count"],
                "damage_summary": detection_result["damage_summary"],
                "counts": detection_result["counts"],
                "priority": detection_result["priority"],
                "priority_color": detection_result["priority_color"],
                "urgency_note": detection_result["urgency_note"],
                "detections": detection_result["detections"],
            },
            "location": {
                "lat": lat,
                "lon": lon,
                "address_display": address_display,
                "geo_details": geo_data,
            },
            "municipality": muni_info,
            "draft_email": draft,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")


@app.post("/api/send-report")
def send_report_endpoint(req: SendReportRequest):
    """Dispatches grievance email with attached annotated road photo."""
    img_bytes = None
    if req.image_base64:
        raw_b64 = req.image_base64
        if "base64," in raw_b64:
            raw_b64 = raw_b64.split("base64,")[1]
        try:
            img_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            print(f"[SendReport] Failed to decode image_base64: {e}")

    result = send_grievance_email(
        to_email=req.to_email,
        subject=req.subject,
        body_text=req.body,
        annotated_image_bytes=img_bytes,
        cc_citizen_email=req.cc_citizen_email,
        citizen_name=req.citizen_name,
    )

    return result
