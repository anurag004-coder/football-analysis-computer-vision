"""Pitch Calibration Tools: Interactive and Preset Keypoint Selection.

Allows users to pick pitch keypoints or load standard broadcast presets
to compute the homography matrix.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.pitch.pitch_model import PitchModel
from src.pitch.homography import PitchHomography


class PitchCalibrator:
    """Calibrator for establishing homography mapping between image frame and pitch."""

    def __init__(self, pitch_model: Optional[PitchModel] = None):
        self.pitch_model = pitch_model or PitchModel()

    def calibrate_from_matched_pairs(
        self,
        image_points: List[Tuple[float, float]],
        keypoint_names: List[str]
    ) -> Tuple[PitchHomography, float]:
        """Compute homography from image pixel points and their pitch keypoint names.

        Args:
            image_points: List of (u, v) pixel points clicked or detected on frame.
            keypoint_names: List of landmark names defined in PitchModel.

        Returns:
            Tuple of (PitchHomography object, reprojection error in pixels).
        """
        if len(image_points) != len(keypoint_names):
            raise ValueError("Number of image points must match number of keypoint names.")

        pitch_points = [self.pitch_model.get_keypoint(name) for name in keypoint_names]

        homography = PitchHomography()
        H, err = homography.compute_from_points(image_points, pitch_points)
        return homography, err

    def create_broadcast_preset(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080
    ) -> PitchHomography:
        """Create a default homography preset for standard high-sideline broadcast camera.

        Covers standard wide camera showing center circle and penalty box.
        Useful when automated or manual clicks are skipped.
        """
        # Standard broadcast perspective mapping approximation
        # (Assuming camera centered around midfield, showing approx. x in [15, 90] m)
        w, h = frame_width, frame_height
        img_pts = [
            (0.12 * w, 0.28 * h),   # Top-left field area
            (0.88 * w, 0.28 * h),   # Top-right field area
            (0.98 * w, 0.95 * h),   # Bottom-right sideline
            (0.02 * w, 0.95 * h),   # Bottom-left sideline
            (0.50 * w, 0.60 * h),   # Midfield center spot
        ]
        pitch_pts = [
            (15.0, 0.0),
            (90.0, 0.0),
            (95.0, 68.0),
            (10.0, 68.0),
            (52.5, 34.0),
        ]
        homography = PitchHomography()
        homography.compute_from_points(img_pts, pitch_pts)
        return homography

    def interactive_calibrate(
        self,
        frame: np.ndarray,
        keypoints_to_select: Optional[List[str]] = None
    ) -> Tuple[PitchHomography, float, Dict[str, Tuple[float, float]]]:
        """Launch interactive OpenCV window to select pitch landmarks on a frame.

        Args:
            frame: BGR image frame from the video.
            keypoints_to_select: List of landmark names to select. Defaults to 4 key landmarks.

        Returns:
            Tuple of (PitchHomography, reprojection_error, dict of landmark -> (x, y)).
        """
        if keypoints_to_select is None:
            keypoints_to_select = [
                "halfway_top",
                "center_spot",
                "halfway_bottom",
                "left_pen_box_top_right",
                "left_pen_box_bot_right",
            ]

        collected_points: List[Tuple[float, float]] = []
        current_idx = [0]
        window_name = "Select Pitch Keypoints (Press 'r' to reset, 'q' when done)"
        display_copy = frame.copy()

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and current_idx[0] < len(keypoints_to_select):
                collected_points.append((float(x), float(y)))
                kp_name = keypoints_to_select[current_idx[0]]
                cv2.circle(display_copy, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    display_copy,
                    f"{current_idx[0] + 1}: {kp_name}",
                    (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                current_idx[0] += 1

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_cb)

        while True:
            canvas = display_copy.copy()
            if current_idx[0] < len(keypoints_to_select):
                prompt = f"Click point {current_idx[0] + 1}/{len(keypoints_to_select)}: {keypoints_to_select[current_idx[0]]}"
                cv2.putText(canvas, prompt, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                prompt = "All points selected! Press 'c' to compute homography or 'r' to reset."
                cv2.putText(canvas, prompt, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                collected_points.clear()
                current_idx[0] = 0
                display_copy = frame.copy()
            elif key == ord('c') or key == ord('q') or key == 27:
                if len(collected_points) >= 4:
                    break

        cv2.destroyWindow(window_name)

        selected_names = keypoints_to_select[:len(collected_points)]
        homography, err = self.calibrate_from_matched_pairs(collected_points, selected_names)
        point_dict = {name: pt for name, pt in zip(selected_names, collected_points)}
        return homography, err, point_dict
