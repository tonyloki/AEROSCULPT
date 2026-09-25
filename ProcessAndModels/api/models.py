"""
Pydantic data models and schemas for the Aerosculpt system.
"""
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "PENDING"
    INGESTING = "INGESTING"
    KEYFRAME_FILTERING = "KEYFRAME_FILTERING"
    SCENE_UNDERSTANDING = "SCENE_UNDERSTANDING"
    SPARSE_RECONSTRUCTION = "SPARSE_RECONSTRUCTION"
    DENSE_MVS = "DENSE_MVS"
    GEOREFERENCING = "GEOREFERENCING"
    METRIC_VALIDATION = "METRIC_VALIDATION"
    EXPORTING = "EXPORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ValidationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    FLAGGED = "FLAGGED"


class TelemetryPoint(BaseModel):
    timestamp_sec: float
    frame_index: Optional[int] = None
    latitude: float
    longitude: float
    altitude_msl: float
    altitude_rel: float
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    rtk_fix: bool = False
    accuracy_horizontal: float = 1.0  # In meters
    accuracy_vertical: float = 1.5


class KeyframeResult(BaseModel):
    frame_id: int
    timestamp_sec: float
    image_path: str
    blur_laplacian: float
    exposure_mean: float
    is_keyframe: bool
    dynamic_ratio: float = 0.0
    mask_path: Optional[str] = None


class ReconstructionOutputs(BaseModel):
    textured_mesh_glb: Optional[str] = None
    textured_mesh_obj: Optional[str] = None
    dense_pointcloud_ply: Optional[str] = None
    pointcloud_las: Optional[str] = None
    orthophoto_geotiff: Optional[str] = None
    dsm_geotiff: Optional[str] = None
    camera_trajectory_geojson: Optional[str] = None
    bounding_footprint_geojson: Optional[str] = None
    evidence_map_json: Optional[str] = None


class ReconstructionMetrics(BaseModel):
    total_frames_ingested: int = 0
    keyframes_selected: int = 0
    sparse_points_count: int = 0
    dense_points_count: int = 0
    mesh_triangles_count: int = 0
    mean_reprojection_error_px: float = 0.0
    spatial_accuracy_meters: float = 0.0
    rtk_alignment_residual_cm: float = 0.0
    processing_duration_seconds: float = 0.0
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    target_processing_met: bool = True
    target_accuracy_met: bool = True
    utm_zone: str = "UTM 43N"
    datum_elevation_base: float = 0.0


class PipelineJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_percent: float
    current_stage: str
    stage_progress: Dict[str, float] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    metrics: Optional[ReconstructionMetrics] = None
    outputs: Optional[ReconstructionOutputs] = None
    error_message: Optional[str] = None


class MeasureRequest(BaseModel):
    point_a: List[float] = Field(..., description="[X, Y, Z] in metric coordinates")
    point_b: List[float] = Field(..., description="[X, Y, Z] in metric coordinates")
    utm_zone: Optional[str] = None


class MeasureResponse(BaseModel):
    euclidean_distance_m: float
    horizontal_distance_m: float
    vertical_height_diff_m: float
    slope_degrees: float
    confidence_score: float


class CoordinateTransformRequest(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    source_crs: str = "EPSG:4326"
    target_crs: str = "EPSG:32643"


class CoordinateTransformResponse(BaseModel):
    easting_x: float
    northing_y: float
    elevation_z: float
    target_crs: str
