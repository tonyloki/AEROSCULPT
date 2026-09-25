"""
Unified End-to-End UAV 3D Reconstruction Pipeline Orchestrator.
Orchestrates the entire flow from single-pass drone video + telemetry to georeferenced 3D world.
"""
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np

from backend.api.models import (
    JobStatus,
    PipelineJobResponse,
    ReconstructionMetrics,
    ReconstructionOutputs,
    ValidationStatus
)
from backend.config import GLOBAL_CONFIG, OUTPUT_DIR, UPLOAD_DIR
from backend.core.video_ingest import TelemetryParser, VideoIngestEngine
from backend.core.keyframe_engine import KeyframeEngine
from backend.core.scene_segmentation import SceneUnderstandingEngine
from backend.core.motion_priors import MotionPriorsEngine
from backend.core.adaptive_sfm import AdaptiveSfMEngine
from backend.core.dense_mvs import DenseMVSEngine
from backend.core.georeferencing import GeoreferencingEngine
from backend.core.metric_validation import MetricValidationEngine
from backend.core.gis_exporter import GISExporter


class AeroSculptPipeline:
    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id or f"flight_{uuid.uuid4().hex[:8]}"
        self.job_dir = OUTPUT_DIR / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate modular engines
        self.keyframe_engine = KeyframeEngine()
        self.scene_engine = SceneUnderstandingEngine()
        self.sfm_engine = AdaptiveSfMEngine()
        self.mvs_engine = DenseMVSEngine(self.job_dir)
        self.georef_engine = GeoreferencingEngine()
        self.validation_engine = MetricValidationEngine()
        self.gis_exporter = GISExporter(self.job_dir)

    def execute(
        self,
        video_path: Path,
        telemetry_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> PipelineJobResponse:
        """
        Executes end-to-end 3D reconstruction pipeline.
        Progress callback signature: (stage_name, progress_percent, status_message)
        """
        start_time = time.time()
        logs: List[str] = []

        def log(msg: str):
            logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        def update(stage: str, pct: float, msg: str):
            log(f"Stage: {stage} ({pct:.0f}%) - {msg}")
            if progress_callback:
                progress_callback(stage, pct, msg)

        try:
            # ----------------------------------------------------
            # STAGE 1: UAV INPUT & TELEMETRY INGESTION
            # ----------------------------------------------------
            update("UAV_INPUT", 10.0, "Ingesting UAV video and flight telemetry...")
            ingest_engine = VideoIngestEngine(video_path, self.job_dir)
            video_meta = ingest_engine.extract_metadata()
            log(f"Video resolution: {video_meta['width']}x{video_meta['height']} @ {video_meta['fps']:.1f} fps ({video_meta['total_frames']} total frames)")

            # Parse Telemetry
            telemetry_points = []
            if telemetry_path and telemetry_path.exists():
                if telemetry_path.suffix.lower() == ".srt":
                    telemetry_points = TelemetryParser.parse_dji_srt(telemetry_path)
                elif telemetry_path.suffix.lower() == ".json":
                    telemetry_points = TelemetryParser.parse_json_telemetry(telemetry_path)
                log(f"Parsed {len(telemetry_points)} raw telemetry waypoints.")

            # Sample frames for keyframe analysis
            sampled_frames = ingest_engine.sample_frames(sample_rate_fps=2.0)
            telemetry_map = ingest_engine.interpolate_telemetry(sampled_frames, telemetry_points)
            log(f"Sampled {len(sampled_frames)} preliminary frame candidates.")

            # ----------------------------------------------------
            # STAGE 2: QUALITY & INTELLIGENT KEYFRAME FILTERING
            # ----------------------------------------------------
            update("KEYFRAME_FILTERING", 25.0, "Applying blur, exposure and viewpoint disparity filtering...")
            keyframe_results = self.keyframe_engine.process_candidates(sampled_frames)
            selected_keyframes = [k for k in keyframe_results if k.is_keyframe]
            log(f"Filtered {len(sampled_frames)} frames down to {len(selected_keyframes)} high-value keyframes.")

            # ----------------------------------------------------
            # STAGE 3: AI SCENE & MOTION UNDERSTANDING
            # ----------------------------------------------------
            update("SCENE_UNDERSTANDING", 40.0, "Generating dynamic object instance masks and classifying scene complexity...")
            dynamic_masks: Dict[int, np.ndarray] = {}
            scene_complexities = []

            for kf in selected_keyframes:
                import cv2
                img = cv2.imread(kf.image_path)
                if img is not None:
                    mask, dyn_ratio, _ = self.scene_engine.generate_dynamic_mask(img)
                    dynamic_masks[kf.frame_id] = mask
                    kf.dynamic_ratio = dyn_ratio
                    complexity, _ = self.scene_engine.classify_scene_complexity(img, dyn_ratio)
                    scene_complexities.append(complexity)

            # Dominant scene complexity
            complexity_mode = max(set(scene_complexities), key=scene_complexities.count) if scene_complexities else "MEDIUM"
            log(f"Scene complexity assessed: {complexity_mode}. Dynamic object masks generated for SfM exclusion.")

            # ----------------------------------------------------
            # STAGE 4: ADAPTIVE 3D RECONSTRUCTION (GLUEMAP / SfM)
            # ----------------------------------------------------
            update("SPARSE_RECONSTRUCTION", 55.0, f"Running adaptive SfM ({complexity_mode}) with GNSS priors...")
            sfm_result = self.sfm_engine.run_reconstruction(
                selected_keyframes, dynamic_masks, telemetry_map, complexity_mode=complexity_mode
            )
            sparse_pts = sfm_result["sparse_points"]
            sparse_clrs = sfm_result["sparse_colors"]
            cam_poses = sfm_result["camera_poses"]
            rep_error = sfm_result["mean_reprojection_error"]
            log(f"Triangulated {len(sparse_pts)} sparse points. Mean reprojection error: {rep_error:.3f} px.")

            # ----------------------------------------------------
            # STAGE 5: DENSE MVS & 3D MESH GENERATION
            # ----------------------------------------------------
            update("DENSE_MVS", 70.0, "Densifying point cloud and reconstructing textured 3D mesh...")
            mvs_outputs = self.mvs_engine.process_and_export_all(sparse_pts, sparse_clrs, self.job_id)
            log(f"Generated {mvs_outputs['dense_point_count']} dense points and {mvs_outputs['mesh_triangle_count']} mesh triangles.")

            # ----------------------------------------------------
            # STAGE 6: GEOREFERENCING & COORDINATE ALIGNMENT
            # ----------------------------------------------------
            update("GEOREFERENCING", 82.0, "Solving Sim(3) 7-DOF spatial alignment and CRS transformation...")
            georef_result = self.georef_engine.align_to_world_crs(cam_poses, telemetry_map)
            log(f"Spatial alignment complete. CRS: {georef_result['target_crs']}, RMSE: {georef_result['rmse_meters']:.3f} m, RTK residual: {georef_result['rtk_residual_cm']:.1f} cm.")

            # ----------------------------------------------------
            # STAGE 7: GIS PRODUCTS EXPORT
            # ----------------------------------------------------
            update("EXPORTING", 90.0, "Exporting GIS GeoTIFF (DSM) and GeoJSON flight trajectory...")
            ref_lat = telemetry_map[selected_keyframes[0].frame_id].latitude if selected_keyframes else 13.0827
            ref_lon = telemetry_map[selected_keyframes[0].frame_id].longitude if selected_keyframes else 80.2707

            traj_geojson = self.gis_exporter.export_flight_trajectory_geojson(telemetry_map, self.job_id)
            footprint_geojson = self.gis_exporter.export_survey_footprint_geojson(telemetry_map, self.job_id)
            dsm_geotiff = self.gis_exporter.export_geotiff_dsm(sparse_pts, ref_lat, ref_lon, self.job_id)
            log(f"GIS datasets saved to {self.job_dir}")

            # ----------------------------------------------------
            # STAGE 8: METRIC VALIDATION FIRST
            # ----------------------------------------------------
            update("METRIC_VALIDATION", 96.0, "Performing geometric consistency and metric validation checks...")
            total_duration = time.time() - start_time
            val_result = self.validation_engine.validate_reconstruction(
                mean_reprojection_error=rep_error,
                georef_rmse_meters=georef_result["rmse_meters"],
                rtk_residual_cm=georef_result["rtk_residual_cm"],
                has_rtk=georef_result.get("has_rtk", False),
                point_count=mvs_outputs["dense_point_count"],
                duration_seconds=total_duration
            )
            log(f"Validation Status: {val_result['status'].value}. Accuracy Met: {val_result['target_accuracy_met']}.")

            # Assemble complete outputs
            outputs = ReconstructionOutputs(
                textured_mesh_glb=mvs_outputs["textured_glb"],
                textured_mesh_obj=mvs_outputs["textured_obj"],
                dense_pointcloud_ply=mvs_outputs["dense_ply"],
                pointcloud_las=mvs_outputs["dense_ply"].replace(".ply", ".las"),
                dsm_geotiff=str(dsm_geotiff),
                camera_trajectory_geojson=str(traj_geojson),
                bounding_footprint_geojson=str(footprint_geojson),
                evidence_map_json=mvs_outputs["evidence_json"]
            )

            metrics = ReconstructionMetrics(
                total_frames_ingested=video_meta["total_frames"],
                keyframes_selected=len(selected_keyframes),
                sparse_points_count=len(sparse_pts),
                dense_points_count=mvs_outputs["dense_point_count"],
                mesh_triangles_count=mvs_outputs["mesh_triangle_count"],
                mean_reprojection_error_px=round(rep_error, 3),
                spatial_accuracy_meters=round(georef_result["rmse_meters"], 3),
                rtk_alignment_residual_cm=round(georef_result["rtk_residual_cm"], 1),
                processing_duration_seconds=round(total_duration, 1),
                validation_status=val_result["status"],
                target_processing_met=val_result["target_processing_met"],
                target_accuracy_met=val_result["target_accuracy_met"],
                utm_zone=georef_result["target_crs"]
            )

            update("COMPLETED", 100.0, "3D World Generation Successfully Completed.")
            log(f"Reconstruction finished in {total_duration:.1f} seconds.")

            return PipelineJobResponse(
                job_id=self.job_id,
                status=JobStatus.COMPLETED,
                progress_percent=100.0,
                current_stage="COMPLETED",
                logs=logs,
                metrics=metrics,
                outputs=outputs
            )

        except Exception as e:
            import traceback
            err_details = traceback.format_exc()
            log(f"Pipeline failure: {str(e)}\n{err_details}")
            return PipelineJobResponse(
                job_id=self.job_id,
                status=JobStatus.FAILED,
                progress_percent=0.0,
                current_stage="FAILED",
                logs=logs,
                error_message=str(e)
            )
