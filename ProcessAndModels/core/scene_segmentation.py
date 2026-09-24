"""
Semantic Scene Understanding and Dynamic Object Masking Engine.
Employs YOLO Segmentation to mask out dynamic foreground objects (vehicles, pedestrians)
and evaluates scene complexity (EASY -> MEDIUM -> HARD) to guide adaptive reconstruction.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

from backend.config import SceneUnderstandingConfig, GLOBAL_CONFIG


class SceneUnderstandingEngine:
    def __init__(self, config: SceneUnderstandingConfig = GLOBAL_CONFIG.scene):
        self.config = config
        self._model = None

    def _get_model(self):
        """Lazy loader for YOLO segmentation model."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.config.model_name)
            except Exception:
                # Fallback to simulated semantic detector if torch/weights not downloaded
                self._model = "MOCK_SEGMENTATION"
        return self._model

    def generate_dynamic_mask(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float, List[Dict]]:
        """
        Detects dynamic objects (cars, pedestrians, trucks, boats) and produces a dilated binary mask.
        Returns:
            binary_mask: 2D uint8 array (255 for dynamic pixels, 0 for static background)
            dynamic_ratio: fraction of image covered by dynamic objects (0.0 to 1.0)
            detections: list of detected dynamic objects with bounding boxes
        """
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        model = self._get_model()
        detections = []

        if model != "MOCK_SEGMENTATION":
            try:
                results = model.predict(
                    image_bgr,
                    conf=self.config.confidence_threshold,
                    classes=self.config.dynamic_classes,
                    verbose=False
                )
                if results and len(results) > 0 and results[0].masks is not None:
                    masks_data = results[0].masks.data.cpu().numpy()
                    for idx, single_mask in enumerate(masks_data):
                        resized_mask = cv2.resize(single_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask = np.maximum(mask, (resized_mask * 255).astype(np.uint8))
                        
                        box = results[0].boxes[idx]
                        detections.append({
                            "class_id": int(box.cls[0].item()),
                            "confidence": float(box.conf[0].item()),
                            "bbox": [float(x) for x in box.xyxy[0].tolist()]
                        })
            except Exception:
                pass
        else:
            # Fallback heuristic: motion/color edge anomaly detector for demo/offline test
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            # Find localized high-contrast small blobs (simulating cars/people)
            edges = cv2.Canny(gray, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < (w * h * 0.05):
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = bw / float(bh)
                    if 0.3 < aspect < 3.0:
                        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

        # Dilate mask to ensure boundaries and shadows of dynamic objects are completely excluded
        if np.any(mask):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.dilation_kernel_size, self.config.dilation_kernel_size)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        dynamic_ratio = float(np.sum(mask > 0) / (h * w))
        return mask, dynamic_ratio, detections

    def classify_scene_complexity(
        self, image_bgr: np.ndarray, dynamic_ratio: float
    ) -> Tuple[str, Dict[str, float]]:
        """
        Assesses scene complexity based on:
        - Texture density (Laplacian energy & Harris corner distribution)
        - Edge richness (Canny edge percentage)
        - Dynamic object occlusion ratio
        
        Outputs: "EASY", "MEDIUM", or "HARD"
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Texture richness
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        num_corners = len(corners) if corners is not None else 0
        corner_density = num_corners / 1000.0

        # 2. Edge percentage
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges) / (h * w))

        # 3. Overall complexity index
        # Repetitive texture or extreme dynamic occlusion -> HARD
        # Medium urban landscape with standard feature trackability -> MEDIUM
        # Clear open landscape / well-textured infrastructure -> EASY
        complexity_score = (corner_density * 0.4) + (edge_ratio * 3.0) + (dynamic_ratio * 1.5)

        if dynamic_ratio > 0.25 or corner_density < 0.15:
            classification = "HARD"
        elif complexity_score > 0.45 or dynamic_ratio > 0.08:
            classification = "MEDIUM"
        else:
            classification = "EASY"

        metrics = {
            "corner_density": float(corner_density),
            "edge_ratio": float(edge_ratio),
            "dynamic_ratio": float(dynamic_ratio),
            "complexity_score": float(complexity_score)
        }

        return classification, metrics
