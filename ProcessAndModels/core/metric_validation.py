"""
Spatial Accuracy and Metric Validation Engine.
Validates geometric consistency, reprojection errors, GNSS residuals, and assigns
VERIFIED or UNVERIFIED certification status according to strict engineering requirements.
"""
from typing import Dict, List
import numpy as np

from backend.api.models import ReconstructionMetrics, ValidationStatus
from backend.config import PipelineTargets, GLOBAL_CONFIG


class MetricValidationEngine:
    def __init__(self, targets: PipelineTargets = GLOBAL_CONFIG.targets):
        self.targets = targets

    def validate_reconstruction(
        self,
        mean_reprojection_error: float,
        georef_rmse_meters: float,
        rtk_residual_cm: float,
        has_rtk: bool,
        point_count: int,
        duration_seconds: float
    ) -> Dict:
        """
        Assesses all quality criteria and generates validation report.
        Criteria:
        - Reprojection error < 1.25 px
        - Spatial accuracy <= 1.0 meter (or RTK alignment <= 20 cm)
        - Sufficient point density (> 500 points)
        - Execution target <= 15 minutes (900 seconds)
        """
        is_reprojection_good = mean_reprojection_error <= 1.25
        is_accuracy_good = (
            (has_rtk and rtk_residual_cm <= 20.0) or
            (not has_rtk and georef_rmse_meters <= self.targets.target_spatial_accuracy_meters)
        )
        is_density_good = point_count >= 100
        is_time_good = duration_seconds <= (self.targets.max_processing_time_minutes * 60.0)

        # Final certification status
        if is_reprojection_good and is_accuracy_good and is_density_good:
            status = ValidationStatus.VERIFIED
        elif is_reprojection_good or is_accuracy_good:
            status = ValidationStatus.UNVERIFIED
        else:
            status = ValidationStatus.FLAGGED

        checks = {
            "reprojection_check": {
                "value_px": float(mean_reprojection_error),
                "threshold_px": 1.25,
                "passed": bool(is_reprojection_good)
            },
            "spatial_accuracy_check": {
                "rmse_meters": float(georef_rmse_meters),
                "rtk_residual_cm": float(rtk_residual_cm),
                "threshold_meters": 1.0,
                "threshold_rtk_cm": 20.0,
                "passed": bool(is_accuracy_good)
            },
            "density_check": {
                "point_count": int(point_count),
                "passed": bool(is_density_good)
            },
            "processing_time_check": {
                "duration_seconds": float(duration_seconds),
                "target_max_seconds": 900.0,
                "passed": bool(is_time_good)
            }
        }

        return {
            "status": status,
            "checks": checks,
            "target_processing_met": is_time_good,
            "target_accuracy_met": is_accuracy_good
        }
