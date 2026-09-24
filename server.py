"""
Production FastAPI REST API Server for Face-motion-gpu.
Keeps multimodal AI models resident in GPU VRAM (Warm State) to eliminate cold-start latency.
Exposes endpoints for external microservices, web apps, and mobile clients.
"""

import os
import sys
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from service import MultimodalEvaluatorService
from schemas_v2 import AnalyzeRequestV2, BehavioralEvidenceResponse, AnalyzeAutoRequest

# Global service handle
evaluator_service: Optional[MultimodalEvaluatorService] = None
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up and keep models in GPU VRAM on server startup."""
    global evaluator_service
    device = os.environ.get("DEVICE", "cuda")
    print(f"[FastAPI Server] Warming up MultimodalEvaluatorService on '{device}'...")
    evaluator_service = MultimodalEvaluatorService.get_instance(device=device)
    print("[FastAPI Server] Server is warm and ready to receive inference requests!")
    yield
    print("[FastAPI Server] Shutting down server...")


app = FastAPI(
    title="Face-Motion-GPU Multimodal Sale AI API",
    description="High-Throughput Offline Multimodal Sale AI Assessment API (SCRFD + EmotiEffLib + 6DRepNet + Gaze-LLE + SileroVAD + Zipformer)",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for external frontend or microservice integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve annotated videos and markdown reports
app.mount("/static/results", StaticFiles(directory=RESULTS_DIR), name="results")

DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
if os.path.exists(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


class AnalyzePathRequest(BaseModel):
    video_path: str = Field(..., description="Absolute or relative path to the video file on host/shared storage.")
    audio_path: Optional[str] = Field(None, description="Optional separate audio file path (.wav).")
    session_id: Optional[str] = Field(None, description="Optional custom session identifier.")
    batch_size: int = Field(32, ge=1, le=128, description="GPU batch size (default: 32).")
    step: int = Field(2, ge=1, le=30, description="Frame sample step (e.g. 2, 4, 6).")
    save_video: bool = Field(False, description="Whether to render annotated video with HUD.")
    show_hud: bool = Field(True, description="Overlay behavioral HUD on output video.")


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Face-motion-gpu Multimodal Sale AI API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/api/v1/health"
    }


@app.get("/api/v1/health", tags=["Health"])
def health_check():
    """Checks service readiness and GPU VRAM statistics."""
    cuda_available = False
    gpu_name = None
    vram_used_mb = 0
    vram_total_mb = 0

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
    except Exception:
        pass

    return {
        "status": "healthy",
        "service_warm": evaluator_service is not None,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb
    }


@app.post("/api/v1/analyze", tags=["Inference"])
async def analyze_video(
    file: Optional[UploadFile] = File(None, description="Video file uploaded as multipart/form-data"),
    video_path: Optional[str] = Form(None, description="Path to video file already on disk"),
    audio_path: Optional[str] = Form(None, description="Path to separate audio file on disk"),
    session_id: Optional[str] = Form(None),
    batch_size: int = Form(32),
    step: int = Form(2),
    save_video: bool = Form(False),
    show_hud: bool = Form(True)
):
    """
    Main Multimodal Assessment Endpoint.
    Accepts either a direct file upload OR a path to an existing video file on disk.
    Returns complete behavioral breakdown, conversational turns, 1s timeline, and scores.
    """
    if evaluator_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is still initializing or unavailable."
        )

    temp_video_path = None

    try:
        # 1. Handle file upload if provided
        if file is not None:
            suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_video_path = tmp.name
                shutil.copyfileobj(file.file, tmp)
            target_video = temp_video_path
            current_session_id = session_id or os.path.splitext(file.filename or "session")[0]
        elif video_path:
            target_video = video_path
            current_session_id = session_id or os.path.splitext(os.path.basename(video_path))[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' (multipart upload) or 'video_path' (form param) must be provided."
            )

        if not os.path.exists(target_video):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video path does not exist: {target_video}"
            )

        # 2. Run multimodal evaluation (models already warm in VRAM)
        result = evaluator_service.evaluate(
            video_path=target_video,
            audio_path=audio_path,
            output_dir=RESULTS_DIR,
            session_id=current_session_id,
            batch_size=batch_size,
            step=step,
            save_video=save_video,
            show_hud=show_hud,
            show_progress=False
        )

        # 3. Add static URLs for artifacts if available
        if result["files"].get("annotated_video"):
            annotated_filename = os.path.basename(result["files"]["annotated_video"])
            result["files"]["annotated_video_url"] = f"/static/results/{annotated_filename}"

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )
    finally:
        # Clean up temporary uploaded file if any
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass


@app.post("/api/v1/analyze-json", tags=["Inference"])
async def analyze_video_json(payload: AnalyzePathRequest):
    """JSON body alternative for backend microservices passing local/shared storage paths."""
    if evaluator_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is still initializing or unavailable."
        )

    if not os.path.exists(payload.video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found: {payload.video_path}"
        )

    try:
        session_id = payload.session_id or os.path.splitext(os.path.basename(payload.video_path))[0]
        result = evaluator_service.evaluate(
            video_path=payload.video_path,
            audio_path=payload.audio_path,
            output_dir=RESULTS_DIR,
            session_id=session_id,
            batch_size=payload.batch_size,
            step=payload.step,
            save_video=payload.save_video,
            show_hud=payload.show_hud,
            show_progress=False
        )

        if result["files"].get("annotated_video"):
            annotated_filename = os.path.basename(result["files"]["annotated_video"])
            result["files"]["annotated_video_url"] = f"/static/results/{annotated_filename}"

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.post("/api/v2/analyze", tags=["Inference v2"], response_model=BehavioralEvidenceResponse)
async def analyze_behavioral_evidence(payload: AnalyzeRequestV2):
    """
    Sensory Engine v2: Extracts objective multimodal behavioral evidence
    (Eye contact %, Nodding count, Speech WPM, Hesitations, Turn-by-turn evidence)
    for consumption by LLMs or downstream analytics.
    Supports both 'video' and 'audio' (when webcam is disabled).
    """
    if evaluator_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is still initializing or unavailable."
        )

    if not os.path.exists(payload.video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media file not found: {payload.video_path}"
        )

    try:
        evidence = evaluator_service.extract_behavioral_evidence(
            video_path=payload.video_path,
            turn_markers=payload.turn_markers,
            media_type=payload.media_type,
            step=payload.step,
            batch_size=payload.batch_size,
            session_id=payload.session_id,
            include_timeline_1s=payload.include_timeline_1s
        )
        return JSONResponse(content=evidence)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Behavioral evidence extraction failed: {str(e)}"
        )


@app.post("/api/v2/analyze-auto", tags=["Inference v2 (Auto Turns)"], response_model=BehavioralEvidenceResponse)
async def analyze_auto_json(payload: AnalyzeAutoRequest):
    """
    Sensory Engine v2 (Auto Turns - JSON Body):
    Directly assesses video without requiring pre-annotated turn markers.
    Uses Zipformer ASR for Vietnamese transcription and Silero VAD (default threshold=0.3).
    Splits into a separate speaker turn whenever speech pauses for >= pause_threshold_sec (default: 0.3s).
    All turns are pure speaker turns (role: 'speaker', no artificial student/ai split).
    Returns complete behavioral evidence JSON matching /api/v2/analyze.
    """
    if evaluator_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is still initializing or unavailable."
        )

    if not os.path.exists(payload.video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media file not found: {payload.video_path}"
        )

    try:
        evidence = evaluator_service.extract_behavioral_evidence(
            video_path=payload.video_path,
            turn_markers=None,
            media_type="video",
            step=payload.step,
            batch_size=payload.batch_size,
            session_id=payload.session_id,
            include_timeline_1s=payload.include_timeline_1s,
            vad_threshold=payload.vad_threshold,
            pause_threshold_sec=payload.pause_threshold_sec
        )
        return JSONResponse(content=evidence)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Auto behavioral evidence extraction failed: {str(e)}"
        )


@app.post("/api/v2/analyze-video", tags=["Inference v2 (Auto Turns)"], response_model=BehavioralEvidenceResponse)
async def analyze_video_auto_upload(
    file: Optional[UploadFile] = File(None, description="Video file uploaded as multipart/form-data"),
    video_path: Optional[str] = Form(None, description="Path to video file already on disk"),
    session_id: Optional[str] = Form(None),
    vad_threshold: float = Form(0.3, description="VAD sensitivity threshold (default: 0.3)"),
    pause_threshold_sec: float = Form(0.3, description="Silence pause (seconds) to cut a new turn (default: 0.3s)"),
    step: int = Form(6, description="Frame sampling step (default: 6)"),
    batch_size: int = Form(32, description="GPU batch size (default: 32)"),
    include_timeline_1s: bool = Form(True, description="Include 1s resolution timeline")
):
    """
    Sensory Engine v2 (Auto Turns - File Upload / Form Path):
    Upload video file directly OR specify an existing video path on the server.
    Uses Zipformer ASR + sensitive Silero VAD (0.3).
    Splits into separate speaker turns whenever the user pauses for >= pause_threshold_sec (default: 0.3s).
    All turns are pure speaker turns (role: 'speaker', no artificial student/ai split).
    Returns complete behavioral evidence JSON matching /api/v2/analyze.
    """
    if evaluator_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is still initializing or unavailable."
        )

    temp_video_path = None

    try:
        if file is not None:
            suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_video_path = tmp.name
                shutil.copyfileobj(file.file, tmp)
            target_video = temp_video_path
            current_session_id = session_id or os.path.splitext(file.filename or "session")[0]
        elif video_path:
            target_video = video_path
            current_session_id = session_id or os.path.splitext(os.path.basename(video_path))[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' (multipart upload) or 'video_path' (form param) must be provided."
            )

        if not os.path.exists(target_video):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video path does not exist: {target_video}"
            )

        evidence = evaluator_service.extract_behavioral_evidence(
            video_path=target_video,
            turn_markers=None,
            media_type="video",
            step=step,
            batch_size=batch_size,
            session_id=current_session_id,
            include_timeline_1s=include_timeline_1s,
            vad_threshold=vad_threshold,
            pause_threshold_sec=pause_threshold_sec
        )

        return JSONResponse(content=evidence)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Auto video behavioral assessment failed: {str(e)}"
        )
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass


@app.post("/api/v2/analyze-upload", tags=["Inference v2 (Upload)"], response_model=BehavioralEvidenceResponse)
async def analyze_video_upload_with_markers(
    file: UploadFile = File(..., description="Video/Audio file uploaded directly as multipart/form-data"),
    session_id: Optional[str] = Form(None),
    media_type: str = Form("video", description="'video' or 'audio'"),
    turn_markers: Optional[str] = Form(None, description="JSON string array of TurnMarker objects"),
    step: int = Form(6, description="Frame sampling step"),
    batch_size: int = Form(32, description="GPU batch size"),
    include_timeline_1s: bool = Form(False, description="Include 1s resolution timeline")
):
    """
    Sensory Engine v2 (Direct File Upload with Turn Markers):
    Allows remote clients/backends (e.g. Couchee Backend on Windows) to upload a video/audio file
    and provide turn_markers as JSON string.
    """
    if evaluator_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is still initializing or unavailable."
        )

    temp_video_path = None
    try:
        suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_video_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        current_session_id = session_id or os.path.splitext(file.filename or "session")[0]

        parsed_markers = None
        if turn_markers:
            try:
                import json
                parsed_markers = json.loads(turn_markers) if isinstance(turn_markers, str) else turn_markers
            except Exception as e:
                print(f"[Warning] Failed to parse turn_markers JSON: {e}")
                parsed_markers = None

        evidence = evaluator_service.extract_behavioral_evidence(
            video_path=temp_video_path,
            turn_markers=parsed_markers,
            media_type=media_type,
            step=step,
            batch_size=batch_size,
            session_id=current_session_id,
            include_timeline_1s=include_timeline_1s
        )

        return JSONResponse(content=evidence)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload behavioral assessment failed: {str(e)}"
        )
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1)

