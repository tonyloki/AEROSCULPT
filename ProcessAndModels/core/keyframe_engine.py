"""
AI-Driven Keyframe Selection Engine.
Selects informative, blur-free, well-exposed, non-redundant keyframes from UAV video streams.
Filters ~18,000 raw frames down to ~500-600 high-value viewpoints for SfM/MVS.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

from backend.api.models import KeyframeResult
from backend.config import KeyframeConfig, GLOBAL_CONFIG


class KeyframeEngine:
    def __init__(self, config: KeyframeConfig = GLOBAL_CONFIG.keyframe):
        self.config = config

    @staticmethod
    def calculate_laplacian_blur(image_gray: np.ndarray) -> float:
        """Computes the variance of the Laplacian to quantify sharpness."""
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def calculate_fft_blur(image_gray: np.ndarray) -> float:
        """Computes frequency domain energy to detect directional motion blur."""
        h, w = image_gray.shape
        cy, cx = h // 2, w // 2
        f = np.fft.fft2(image_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
        # Low frequency center mask
        mask_radius = min(h, w) // 10
        magnitude_spectrum[cy - mask_radius : cy + mask_radius, cx - mask_radius : cx + mask_radius] = 0
        mean_high_freq = float(np.mean(magnitude_spectrum))
        return mean_high_freq

    @staticmethod
    def evaluate_exposure(image_gray: np.ndarray) -> Tuple[float, bool]:
        """Calculates luminance mean and determines if under/over-exposed."""
        mean_lum = float(np.mean(image_gray))
        # Histogram check
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        under_ratio = float(np.sum(hist[:25]) / image_gray.size)
        over_ratio = float(np.sum(hist[230:]) / image_gray.size)

        is_valid = (
            GLOBAL_CONFIG.keyframe.underexposure_limit < mean_lum < GLOBAL_CONFIG.keyframe.overexposure_limit
            and under_ratio < 0.40
            and over_ratio < 0.35
        )
        return mean_lum, is_valid

    @staticmethod
    def calculate_perceptual_hash(image_gray: np.ndarray) -> str:
        """Computes an 8x8 DCT-based perceptual hash to filter temporal duplicates."""
        resized = cv2.resize(image_gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:8, :8]
        avg = np.mean(dct_low[1:])  # Exclude DC component
        diff = dct_low > avg
        return "".join(["1" if b else "0" for b in diff.flatten()])

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def evaluate_motion_disparity(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Calculates sparse optical flow disparity between consecutive candidate frames."""
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if corners is None or len(corners) < 10:
            return 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
        if p1 is None or st is None:
            return 0.0

        good_prev = corners[st == 1]
        good_curr = p1[st == 1]
        if len(good_prev) < 5:
            return 0.0

        displacements = np.linalg.norm(good_curr - good_prev, axis=1)
        return float(np.median(displacements))

    def process_candidates(
        self, sampled_frames: List[Tuple[int, float, Path]]
    ) -> List[KeyframeResult]:
        """
        Filters raw candidates into optimal keyframes using blur, exposure, 
        perceptual hash, and optical flow disparity.
        """
        results: List[KeyframeResult] = []
        if not sampled_frames:
            return results

        last_selected_gray = None
        last_selected_hash = ""

        # Phase 1: Quality assessment
        scored_candidates = []
        for frame_idx, t_sec, frame_path in sampled_frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = self.calculate_laplacian_blur(gray)
            fft_score = self.calculate_fft_blur(gray)
            mean_exposure, is_good_exposure = self.evaluate_exposure(gray)

            is_sharp = blur_score >= self.config.laplacian_variance_threshold
            p_hash = self.calculate_perceptual_hash(gray)

            scored_candidates.append({
                "frame_idx": frame_idx,
                "t_sec": t_sec,
                "path": frame_path,
                "gray": gray,
                "blur_score": blur_score,
                "fft_score": fft_score,
                "exposure": mean_exposure,
                "is_sharp": is_sharp,
                "is_exposed": is_good_exposure,
                "hash": p_hash
            })

        # Phase 2: Viewpoint diversity & temporal redundancy reduction
        for item in scored_candidates:
            # Baseline quality filter
            if not item["is_exposed"] and item["blur_score"] < 50.0:
                results.append(
                    KeyframeResult(
                        frame_id=item["frame_idx"],
                        timestamp_sec=item["t_sec"],
                        image_path=str(item["path"]),
                        blur_laplacian=item["blur_score"],
                        exposure_mean=item["exposure"],
                        is_keyframe=False
                    )
                )
                continue

            # First frame always selected
            if last_selected_gray is None:
                last_selected_gray = item["gray"]
                last_selected_hash = item["hash"]
                results.append(
                    KeyframeResult(
                        frame_id=item["frame_idx"],
                        timestamp_sec=item["t_sec"],
                        image_path=str(item["path"]),
                        blur_laplacian=item["blur_score"],
                        exposure_mean=item["exposure"],
                        is_keyframe=True
                    )
                )
                continue

            # Check perceptual difference
            h_dist = self.hamming_distance(item["hash"], last_selected_hash)
            motion_disp = self.evaluate_motion_disparity(last_selected_gray, item["gray"])

            # Must have moved enough in viewpoint and not be a static redundant frame
            is_new_viewpoint = (
                h_dist >= self.config.perceptual_hash_diff_threshold or
                motion_disp >= self.config.min_optical_flow_magnitude
            )

            is_selected = is_new_viewpoint and item["is_sharp"]

            if is_selected:
                last_selected_gray = item["gray"]
                last_selected_hash = item["hash"]

            results.append(
                KeyframeResult(
                    frame_id=item["frame_idx"],
                    timestamp_sec=item["t_sec"],
                    image_path=str(item["path"]),
                    blur_laplacian=item["blur_score"],
                    exposure_mean=item["exposure"],
                    is_keyframe=is_selected
                )
            )

        # Ensure reasonable bounds [min_keyframes, max_keyframes]
        selected_count = sum(1 for r in results if r.is_keyframe)
        if selected_count < self.config.min_keyframes and len(results) > 0:
            # Fallback: pick top sharpest evenly spaced frames
            step = max(1, len(results) // self.config.min_keyframes)
            for i, r in enumerate(results):
                if i % step == 0:
                    r.is_keyframe = True

        return results
