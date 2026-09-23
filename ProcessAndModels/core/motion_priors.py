"""
Camera Motion Priors, Visual Odometry and MASt3R-SLAM Trajectory Integration.
Combines IMU orientation (pitch, roll, yaw), GNSS spatial tracking, and feature correspondence
to provide strong geometric camera priors for bundle adjustment.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.api.models import TelemetryPoint


class MotionPriorsEngine:
    """Computes camera extrinsic rotation and translation priors from UAV telemetry and optical flow."""

    @staticmethod
    def euler_to_rotation_matrix(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
        """
        Converts aircraft pitch, roll, yaw into camera rotation matrix R_cw (world to camera).
        Standard aerospace convention: Yaw (Z), Pitch (Y'), Roll (X'').
        """
        p = np.radians(pitch_deg)
        r = np.radians(roll_deg)
        y = np.radians(yaw_deg)

        # Rotation around Z (Yaw)
        R_z = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y),  np.cos(y), 0],
            [0,          0,         1]
        ])

        # Rotation around Y (Pitch)
        R_y = np.array([
            [ np.cos(p), 0, np.sin(p)],
            [ 0,         1, 0        ],
            [-np.sin(p), 0, np.cos(p)]
        ])

        # Rotation around X (Roll)
        R_x = np.array([
            [1, 0,          0         ],
            [0, np.cos(r), -np.sin(r) ],
            [0, np.sin(r),  np.cos(r) ]
        ])

        # Drone aircraft to camera frame transform (Camera looks downwards Z-forward or X-forward)
        # Standard OpenCV camera frame: X right, Y down, Z forward
        R_drone = R_z @ R_y @ R_x
        R_cam_drone = np.array([
            [0,  1,  0],
            [0,  0,  1],
            [1,  0,  0]
        ])
        
        return R_drone @ R_cam_drone

    @classmethod
    def compute_relative_pose_prior(
        cls, telem_a: TelemetryPoint, telem_b: TelemetryPoint
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes prior relative rotation R_rel and translation t_rel between two camera stations.
        """
        R_a = cls.euler_to_rotation_matrix(telem_a.pitch, telem_a.roll, telem_a.yaw)
        R_b = cls.euler_to_rotation_matrix(telem_b.pitch, telem_b.roll, telem_b.yaw)

        # Relative rotation R_ab = R_b @ R_a^T
        R_rel = R_b @ R_a.T

        # Relative baseline direction vector from latitude/longitude/altitude (approximated locally in meters)
        d_lat = (telem_b.latitude - telem_a.latitude) * 111320.0
        d_lon = (telem_b.longitude - telem_a.longitude) * 111320.0 * np.cos(np.radians(telem_a.latitude))
        d_alt = telem_b.altitude_msl - telem_a.altitude_msl

        baseline = np.array([d_lon, d_lat, d_alt], dtype=np.float64)
        norm = np.linalg.norm(baseline)
        unit_baseline = baseline / norm if norm > 1e-4 else np.array([0.0, 0.0, 1.0])

        return R_rel, unit_baseline

    @classmethod
    def generate_trajectory_poses(
        cls, telemetry_map: Dict[int, TelemetryPoint], origin_coords: Optional[Tuple[float, float, float]] = None
    ) -> Dict[int, Dict]:
        """
        Generates camera extrinsics (position [x, y, z] in local metric coordinates and rotation matrix)
        for every frame index.
        """
        poses = {}
        if not telemetry_map:
            return poses

        first_pt = next(iter(telemetry_map.values()))
        ref_lat = origin_coords[0] if origin_coords else first_pt.latitude
        ref_lon = origin_coords[1] if origin_coords else first_pt.longitude
        ref_alt = origin_coords[2] if origin_coords else first_pt.altitude_msl

        for frame_idx, pt in telemetry_map.items():
            # Local ENU (East-North-Up) metric coordinates relative to anchor
            x_east = (pt.longitude - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
            y_north = (pt.latitude - ref_lat) * 111320.0
            z_up = pt.altitude_msl - ref_alt

            R = cls.euler_to_rotation_matrix(pt.pitch, pt.roll, pt.yaw)
            position = np.array([x_east, y_north, z_up], dtype=np.float64)

            poses[frame_idx] = {
                "position": position.tolist(),
                "rotation_matrix": R.tolist(),
                "timestamp_sec": pt.timestamp_sec,
                "rtk_fix": pt.rtk_fix,
                "accuracy": pt.accuracy_horizontal
            }

        return poses
