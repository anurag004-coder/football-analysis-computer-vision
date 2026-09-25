"""Homography transformation between 2D broadcast camera coordinates and 2D pitch coordinates.

Converts pixel coordinates (feet of players/ball) into metric pitch coordinates (meters).
Validates reprojection errors against PRD requirements (< 3.0 pixels).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class PitchHomography:
    """Manages homography matrix computation and point projection."""

    def __init__(self, homography_matrix: Optional[np.ndarray] = None):
        self.H = homography_matrix
        self.H_inv = None
        if self.H is not None:
            self._update_inverse()

    def _update_inverse(self):
        """Precompute inverse homography matrix."""
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H_inv = None

    def compute_from_points(
        self,
        image_points: Union[np.ndarray, List[Tuple[float, float]]],
        pitch_points: Union[np.ndarray, List[Tuple[float, float]]],
        method: int = cv2.RANSAC,
        ransac_reproj_threshold: float = 5.0,
    ) -> Tuple[np.ndarray, float]:
        """Compute homography matrix H that maps image_points -> pitch_points.

        Args:
            image_points: List or array of (x, y) pixels in broadcast image.
            pitch_points: Corresponding (X, Y) in meters on pitch.
            method: cv2.RANSAC, cv2.LMEDS, or 0.
            ransac_reproj_threshold: RANSAC threshold in pixels.

        Returns:
            Tuple of (homography matrix H, mean reprojection error in pixels).
        """
        img_pts = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
        pitch_pts = np.asarray(pitch_points, dtype=np.float32).reshape(-1, 1, 2)

        if len(img_pts) < 4:
            raise ValueError(f"At least 4 point correspondences required, got {len(img_pts)}")

        H, mask = cv2.findHomography(img_pts, pitch_pts, method, ransac_reproj_threshold)
        if H is None:
            raise RuntimeError("Failed to compute homography matrix from the given points.")

        self.H = H
        self._update_inverse()

        reproj_error = self.evaluate_reprojection_error(image_points, pitch_points)
        return self.H, reproj_error

    def evaluate_reprojection_error(
        self,
        image_points: Union[np.ndarray, List[Tuple[float, float]]],
        pitch_points: Union[np.ndarray, List[Tuple[float, float]]],
    ) -> float:
        """Calculate mean Euclidean distance between original image points and reprojected pitch points."""
        if self.H_inv is None:
            return float("inf")

        img_pts = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        pitch_pts = np.asarray(pitch_points, dtype=np.float32).reshape(-1, 1, 2)

        # Reproject pitch points back into camera image space
        reprojected_img_pts = cv2.perspectiveTransform(pitch_pts, self.H_inv).reshape(-1, 2)
        errors = np.linalg.norm(img_pts - reprojected_img_pts, axis=1)
        mean_error = float(np.mean(errors))
        return mean_error

    def image_to_pitch(self, points: Union[np.ndarray, List[Tuple[float, float]]]) -> np.ndarray:
        """Transform broadcast image pixel coordinates -> pitch metric coordinates (meters).

        Args:
            points: Array or list of shape (N, 2) with (u, v) pixel coordinates.

        Returns:
            Array of shape (N, 2) with (X, Y) in meters.
        """
        if self.H is None:
            raise ValueError("Homography matrix is not set or computed.")

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return transformed

    def pitch_to_image(self, points: Union[np.ndarray, List[Tuple[float, float]]]) -> np.ndarray:
        """Transform pitch metric coordinates (meters) -> broadcast image pixel coordinates.

        Args:
            points: Array or list of shape (N, 2) with (X, Y) meters.

        Returns:
            Array of shape (N, 2) with (u, v) pixel coordinates.
        """
        if self.H_inv is None:
            raise ValueError("Inverse homography matrix is not available.")

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)
        return transformed

    @staticmethod
    def bbox_to_ground_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Extract feet ground contact point from bounding box [x1, y1, x2, y2].

        Feet contact position is bottom-center: ((x1 + x2) / 2, y2).
        """
        x1, y1, x2, y2 = bbox
        return float((x1 + x2) / 2.0), float(y2)

    def save_calibration(self, filepath: Union[str, Path], metadata: Optional[Dict] = None):
        """Save homography matrix and metadata to JSON file."""
        if self.H is None:
            raise ValueError("No homography matrix to save.")

        payload = {
            "homography_matrix": self.H.tolist(),
            "metadata": metadata or {}
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_calibration(cls, filepath: Union[str, Path]) -> "PitchHomography":
        """Load homography matrix from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = np.array(data["homography_matrix"], dtype=np.float32)
        return cls(homography_matrix=matrix)
