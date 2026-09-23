"""
Aerosculpt System Configuration
Defines pipeline parameters, thresholds, CRS definitions, and processing targets.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Directories
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"

for directory in [DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoIngestConfig:
    supported_video_formats: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi")
    supported_telemetry_formats: Tuple[str, ...] = (".srt", ".csv", ".json", ".log")
    max_video_dimension: int = 3840  # 4K UHD support
    target_extraction_fps: float = 2.0  # Initial sample rate prior to keyframing


@dataclass
class KeyframeConfig:
    # Blur detection
    laplacian_variance_threshold: float = 120.0
    fft_blur_threshold: float = 15.0
    # Exposure thresholds (0 - 255)
    underexposure_limit: float = 35.0
    overexposure_limit: float = 225.0
    # Redundancy & Motion Disparity
    min_optical_flow_magnitude: float = 1.8
    perceptual_hash_diff_threshold: int = 8
    target_keyframe_ratio: float = 0.035  # ~600 keyframes from 18,000 frames
    max_keyframes: int = 650
    min_keyframes: int = 50


@dataclass
class SceneUnderstandingConfig:
    model_name: str = "yolov8m-seg.pt"  # Dynamic object instance segmentation
    confidence_threshold: float = 0.40
    # COCO classes for dynamic entities that introduce ghosting/geometric noise
    dynamic_classes: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 5, 7, 8, 15, 16]
    )  # person, bicycle, car, motorcycle, bus, truck, boat, bird, dog
    dilation_kernel_size: int = 15  # Expand dynamic mask boundary
    feather_radius: int = 5


@dataclass
class ReconstructionConfig:
    # Reconstruction strategy: "AUTO", "EASY", "MEDIUM", "HARD"
    strategy: str = "AUTO"
    feature_detector: str = "SIFT_SUPERPOINT"
    matcher: str = "LIGHTGLUE"
    max_features_per_image: int = 4096
    bundle_adjustment_loss: str = "HUBER"
    dense_resolution_divider: int = 1  # 1 = Full resolution, 2 = Half resolution
    poisson_depth: int = 10
    poisson_trim_threshold: float = 7.0


@dataclass
class GeospatialConfig:
    source_crs: str = "EPSG:4326"  # WGS84 geographic
    default_target_crs: str = "EPSG:32643"  # UTM Zone 43N (can be auto-computed from lon/lat)
    elevation_datum: str = "EGM96"
    enable_rtk_ppk_fusion: bool = True
    rtk_accuracy_threshold_meters: float = 0.20
    gnss_standard_accuracy_threshold_meters: float = 1.00
    reprojection_error_threshold_px: float = 1.25


@dataclass
class PipelineTargets:
    max_processing_time_minutes: float = 15.0
    target_spatial_accuracy_meters: float = 1.0
    rtk_spatial_accuracy_meters: float = 0.20
    evidence_states: Tuple[str, ...] = ("observed", "reconstructed", "inferred", "unknown")


@dataclass
class AppConfig:
    video: VideoIngestConfig = field(default_factory=VideoIngestConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    scene: SceneUnderstandingConfig = field(default_factory=SceneUnderstandingConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    geospatial: GeospatialConfig = field(default_factory=GeospatialConfig)
    targets: PipelineTargets = field(default_factory=PipelineTargets)


GLOBAL_CONFIG = AppConfig()
