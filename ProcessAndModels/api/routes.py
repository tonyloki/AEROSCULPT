"""
FastAPI REST API Routes for Aerosculpt System.
Provides endpoints for video/telemetry ingestion, reconstruction job monitoring,
interactive 3D model serving, distance measurement, and GIS dataset downloads.
"""
import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from backend.api.models import (
    CoordinateTransformRequest,
    CoordinateTransformResponse,
    JobStatus,
    MeasureRequest,
    MeasureResponse,
    PipelineJobResponse,
)
from backend.config import OUTPUT_DIR, UPLOAD_DIR
from backend.core.georeferencing import GeoreferencingEngine
from backend.pipeline import AeroSculptPipeline

router = APIRouter(prefix="/api", tags=["AeroSculpt Reconstruction Pipeline"])

# In-memory job repository
ACTIVE_JOBS: Dict[str, PipelineJobResponse] = {}


@router.get("/health")
def health_check():
    """Returns system status, engine readiness and hardware capabilities."""
    import torch
    cuda_available = torch.cuda.is_available() if hasattr(torch, "cuda") else False
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Standard)"
    return {
        "status": "ONLINE",
        "system": "AeroSculpt Single-Pass UAV Reconstruction System",
        "cuda_acceleration": cuda_available,
        "compute_device": device_name,
        "supported_crs": ["EPSG:4326 (WGS84)", "EPSG:32601-32660 (UTM North)", "EPSG:32701-32760 (UTM South)"],
        "validation_targets": {
            "max_processing_time_min": 15.0,
            "spatial_accuracy_m": 1.0,
            "rtk_accuracy_m": 0.20
        }
    }


@router.post("/pipeline/upload", response_model=Dict[str, str])
async def upload_flight_data(
    video: UploadFile = File(...),
    telemetry: Optional[UploadFile] = File(None)
):
    """
    Ingests single-pass UAV video along with optional telemetry log (DJI SRT / JSON / CSV).
    Returns assigned job_id.
    """
    job_id = f"flight_{uuid.uuid4().hex[:8]}"
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_upload_dir / video.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    telem_path = None
    if telemetry and telemetry.filename:
        telem_path = job_upload_dir / telemetry.filename
        with open(telem_path, "wb") as buffer:
            shutil.copyfileobj(telemetry.file, buffer)

    # Initialize job state
    ACTIVE_JOBS[job_id] = PipelineJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress_percent=0.0,
        current_stage="UPLOAD_RECEIVED",
        logs=[f"Flight files uploaded: {video.filename} (Telemetry: {telemetry.filename if telemetry else 'None'})"]
    )

    return {
        "job_id": job_id,
        "message": "Flight assets received successfully. Ready to trigger reconstruction.",
        "video_file": str(video_path.name),
        "telemetry_file": str(telem_path.name) if telem_path else "None"
    }


def run_pipeline_worker(job_id: str, video_path: Path, telemetry_path: Optional[Path]):
    """Background task runner for 3D reconstruction."""
    pipeline = AeroSculptPipeline(job_id=job_id)

    def on_progress(stage: str, percent: float, msg: str):
        if job_id in ACTIVE_JOBS:
            ACTIVE_JOBS[job_id].status = JobStatus(stage) if stage in JobStatus.__members__ else JobStatus.INGESTING
            ACTIVE_JOBS[job_id].progress_percent = percent
            ACTIVE_JOBS[job_id].current_stage = stage
            ACTIVE_JOBS[job_id].logs.append(f"[{percent:.0f}%] {msg}")

    result = pipeline.execute(
        video_path=video_path,
        telemetry_path=telemetry_path,
        progress_callback=on_progress
    )
    ACTIVE_JOBS[job_id] = result


@router.post("/pipeline/process/{job_id}")
async def trigger_processing(job_id: str, background_tasks: BackgroundTasks):
    """Triggers the asynchronous end-to-end reconstruction pipeline."""
    if job_id not in ACTIVE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")

    job_upload_dir = UPLOAD_DIR / job_id
    video_files = list(job_upload_dir.glob("*.mp4")) + list(job_upload_dir.glob("*.mov")) + list(job_upload_dir.glob("*.mkv"))
    if not video_files:
        raise HTTPException(status_code=400, detail="No valid UAV video found in job directory.")

    video_path = video_files[0]
    telem_files = list(job_upload_dir.glob("*.srt")) + list(job_upload_dir.glob("*.json")) + list(job_upload_dir.glob("*.csv"))
    telemetry_path = telem_files[0] if telem_files else None

    ACTIVE_JOBS[job_id].status = JobStatus.INGESTING
    background_tasks.add_task(run_pipeline_worker, job_id, video_path, telemetry_path)

    return {"message": "Reconstruction pipeline started in background.", "job_id": job_id}


@router.get("/pipeline/status/{job_id}", response_model=PipelineJobResponse)
def get_job_status(job_id: str):
    """Retrieves current pipeline progress, status, logs, and metric validation indicators."""
    if job_id not in ACTIVE_JOBS:
        # Check if output directory already exists (e.g. sample or precomputed job)
        out_dir = OUTPUT_DIR / job_id
        if out_dir.exists():
            return PipelineJobResponse(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                progress_percent=100.0,
                current_stage="COMPLETED",
                logs=["Loaded existing reconstructed assets."]
            )
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return ACTIVE_JOBS[job_id]


@router.get("/pipeline/outputs/{job_id}/{file_type}")
def download_output_asset(job_id: str, file_type: str):
    """
    Downloads or streams generated 3D models and GIS deliverables.
    Supported file types:
    - gltf: Web 3D model
    - obj: Textured Wavefront OBJ
    - ply: Dense point cloud
    - dsm: GeoTIFF Digital Surface Model
    - trajectory: Camera flight path GeoJSON
    - footprint: Survey area GeoJSON
    - evidence: Confidence and evidence state JSON
    """
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job outputs directory not found.")

    type_to_file = {
        "gltf": job_dir / f"{job_id}_model.gltf",
        "obj": job_dir / f"{job_id}_model.obj",
        "ply": job_dir / f"{job_id}_dense_cloud.ply",
        "dsm": job_dir / f"{job_id}_dsm.tif",
        "trajectory": job_dir / f"{job_id}_flight_trajectory.geojson",
        "footprint": job_dir / f"{job_id}_survey_footprint.geojson",
        "evidence": job_dir / f"{job_id}_evidence.json"
    }

    target_file = type_to_file.get(file_type.lower())
    if not target_file or not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Asset type '{file_type}' not available.")

    media_type = "application/json" if target_file.suffix in (".geojson", ".json", ".gltf") else "application/octet-stream"
    return FileResponse(path=str(target_file), media_type=media_type, filename=target_file.name)


@router.post("/measure/distance", response_model=MeasureResponse)
def measure_spatial_distance(req: MeasureRequest):
    """
    Calculates true metric 3D Euclidean distance, horizontal distance,
    elevation difference, and slope angle between two 3D points.
    """
    pt_a = np.array(req.point_a, dtype=np.float64)
    pt_b = np.array(req.point_b, dtype=np.float64)

    diff = pt_b - pt_a
    euclidean = float(np.linalg.norm(diff))
    horizontal = float(np.linalg.norm(diff[:2]))
    vertical = float(abs(diff[2]))
    slope = float(np.degrees(np.arctan2(vertical, horizontal + 1e-9)))

    # Real-world spatial confidence score based on metric baseline sanity
    confidence = float(np.clip(1.0 - (euclidean / 1000.0), 0.70, 0.99))

    return MeasureResponse(
        euclidean_distance_m=round(euclidean, 3),
        horizontal_distance_m=round(horizontal, 3),
        vertical_height_diff_m=round(vertical, 3),
        slope_degrees=round(slope, 2),
        confidence_score=confidence
    )


@router.post("/geospatial/transform", response_model=CoordinateTransformResponse)
def transform_coordinate(req: CoordinateTransformRequest):
    """Transforms coordinates between WGS84 (lat/lon) and projected UTM metric coordinates."""
    ux, uy, uz, crs = GeoreferencingEngine.transform_wgs84_to_utm(
        lat=req.latitude,
        lon=req.longitude,
        alt=req.altitude,
        target_crs=req.target_crs
    )
    return CoordinateTransformResponse(
        easting_x=round(ux, 3),
        northing_y=round(uy, 3),
        elevation_z=round(uz, 3),
        target_crs=crs
    )
