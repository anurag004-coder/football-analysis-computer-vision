"""Football Pitch Model and Geometry Definition.

Standard FIFA Pitch Dimensions:
- Length (X-axis): 105.0 meters (0 to 105m)
- Width  (Y-axis):  68.0 meters (0 to 68m)
- Coordinate System: (0, 0) is top-left corner of the pitch.
  X increases left to right. Y increases top to bottom.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class PitchDimensions:
    """Standard pitch dimensions in meters."""
    length: float = 105.0  # X axis
    width: float = 68.0    # Y axis
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    corner_arc_radius: float = 1.0
    goal_width: float = 7.32


class PitchModel:
    """Standardized pitch model with standard reference keypoints."""

    def __init__(self, dimensions: PitchDimensions = None):
        self.dim = dimensions or PitchDimensions()
        self.keypoints = self._compute_reference_keypoints()

    def _compute_reference_keypoints(self) -> Dict[str, Tuple[float, float]]:
        """Compute metric coordinates (X, Y in meters) for standard pitch landmarks."""
        L = self.dim.length
        W = self.dim.width
        p_len = self.dim.penalty_box_length
        p_wid = self.dim.penalty_box_width
        g_len = self.dim.goal_box_length
        g_wid = self.dim.goal_box_width
        p_top = (W - p_wid) / 2.0
        p_bot = (W + p_wid) / 2.0
        g_top = (W - g_wid) / 2.0
        g_bot = (W + g_wid) / 2.0

        kp = {
            # Pitch Outer Corners
            "top_left_corner": (0.0, 0.0),
            "top_right_corner": (L, 0.0),
            "bottom_left_corner": (0.0, W),
            "bottom_right_corner": (L, W),

            # Halfway Line
            "halfway_top": (L / 2.0, 0.0),
            "halfway_bottom": (L / 2.0, W),
            "center_spot": (L / 2.0, W / 2.0),
            "center_circle_top": (L / 2.0, (W / 2.0) - self.dim.center_circle_radius),
            "center_circle_bottom": (L / 2.0, (W / 2.0) + self.dim.center_circle_radius),

            # Left Penalty Box (Home)
            "left_pen_box_top_left": (0.0, p_top),
            "left_pen_box_top_right": (p_len, p_top),
            "left_pen_box_bot_left": (0.0, p_bot),
            "left_pen_box_bot_right": (p_len, p_bot),
            "left_pen_spot": (self.dim.penalty_spot_distance, W / 2.0),

            # Right Penalty Box (Away)
            "right_pen_box_top_left": (L - p_len, p_top),
            "right_pen_box_top_right": (L, p_top),
            "right_pen_box_bot_left": (L - p_len, p_bot),
            "right_pen_box_bot_right": (L, p_bot),
            "right_pen_spot": (L - self.dim.penalty_spot_distance, W / 2.0),

            # Left Goal Box
            "left_goal_box_top_left": (0.0, g_top),
            "left_goal_box_top_right": (g_len, g_top),
            "left_goal_box_bot_left": (0.0, g_bot),
            "left_goal_box_bot_right": (g_len, g_bot),

            # Right Goal Box
            "right_goal_box_top_left": (L - g_len, g_top),
            "right_goal_box_top_right": (L, g_top),
            "right_goal_box_bot_left": (L - g_len, g_bot),
            "right_goal_box_bot_right": (L, g_bot),
        }
        return kp

    def get_keypoint(self, name: str) -> Tuple[float, float]:
        """Return (X, Y) coordinates in meters for keypoint name."""
        if name not in self.keypoints:
            raise KeyError(f"Keypoint '{name}' not found. Available: {list(self.keypoints.keys())}")
        return self.keypoints[name]

    def is_inside_pitch(self, x: float, y: float, margin: float = 3.0) -> bool:
        """Check if coordinates (x, y) in meters fall inside the pitch boundaries."""
        return (-margin <= x <= self.dim.length + margin) and (-margin <= y <= self.dim.width + margin)
