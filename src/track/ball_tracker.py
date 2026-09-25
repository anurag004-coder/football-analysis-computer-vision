"""Football Ball Tracker with Pitch-Masked Contour Heuristics and Kalman/Velocity Smoothing.

Handles small, fast-moving football tracking in broadcast footage where generic
COCO object detectors have low recall on 15-25px balls.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from src.track.tracker import TrackedEntity


class FootballBallTracker:
    """Tracks the football across frames using YOLO detections and pitch-masked heuristics."""

    def __init__(self, max_predict_frames: int = 12, max_speed_px: float = 65.0):
        """Initialize the ball tracker.

        Args:
            max_predict_frames: Maximum frames to extrapolate ball trajectory if temporarily lost.
            max_speed_px: Maximum expected ball displacement between frames (pixels).
        """
        self.max_predict_frames = max_predict_frames
        self.max_speed_px = max_speed_px

        self.last_pos: Optional[Tuple[float, float]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.missing_count: int = 0
        self.last_box: Optional[Tuple[float, float, float, float]] = None

    def _is_inside_player_torso(self, cx: float, cy: float, player_boxes: List[np.ndarray]) -> bool:
        """Check if candidate point is inside a player's upper body (to avoid jersey logos/numbers)."""
        for box in player_boxes:
            x1, y1, x2, y2 = box
            # Check upper 70% of player box
            h = y2 - y1
            if x1 <= cx <= x2 and y1 <= cy <= (y1 + 0.70 * h):
                return True
        return False

    def find_ball_candidate(
        self,
        frame: np.ndarray,
        player_boxes: List[np.ndarray],
    ) -> Optional[Tuple[float, float, float, float]]:
        """Find bright circular ball candidate on the green pitch area."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Green pitch mask (covers standard grass hues)
        pitch_mask = cv2.inRange(hsv, (30, 25, 25), (90, 255, 255))

        # Bright ball candidate mask (intensity > 200)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        cand_mask = cv2.bitwise_and(bright, bright, mask=pitch_mask)

        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cand_mask = cv2.morphologyEx(cand_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 12 <= area <= 380:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / max(1, h)
                if 0.65 <= aspect <= 1.55:
                    perim = cv2.arcLength(cnt, True)
                    if perim > 0:
                        circ = 4 * np.pi * area / (perim * perim)
                        if circ > 0.50:
                            cx = x + w / 2.0
                            cy = y + h / 2.0
                            if not self._is_inside_player_torso(cx, cy, player_boxes):
                                candidates.append((cx, cy, float(w), float(h), area, circ))

        if not candidates:
            return None

        # If previous position exists, pick candidate closest to predicted position
        if self.last_pos is not None:
            pred_x = self.last_pos[0] + self.velocity[0]
            pred_y = self.last_pos[1] + self.velocity[1]
            best_cand = None
            best_dist = float("inf")

            for cx, cy, w, h, area, circ in candidates:
                dist = np.hypot(cx - pred_x, cy - pred_y)
                if dist < self.max_speed_px and dist < best_dist:
                    best_dist = dist
                    best_cand = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

            if best_cand is not None:
                return best_cand

        # Fallback: find candidate with highest circularity and proximity to player ground points
        candidates.sort(key=lambda c: c[4] * c[5], reverse=True)
        cx, cy, w, h, _, _ = candidates[0]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def update(
        self,
        frame: np.ndarray,
        yolo_ball_box: Optional[np.ndarray],
        player_boxes: List[np.ndarray],
        frame_idx: int = 0,
    ) -> Optional[TrackedEntity]:
        """Update ball tracking state and return TrackedEntity for the ball.

        Args:
            frame: Current BGR image frame.
            yolo_ball_box: Bounding box from YOLO if detected [x1, y1, x2, y2].
            player_boxes: Bounding boxes of all detected players.
            frame_idx: Frame index for metadata.

        Returns:
            TrackedEntity with ground contact point and coordinates, or None.
        """
        box = None
        conf = 0.85

        if yolo_ball_box is not None:
            box = tuple(map(float, yolo_ball_box[:4]))
            conf = 0.90
        else:
            box = self.find_ball_candidate(frame, player_boxes)
            conf = 0.70

        if box is not None:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            if self.last_pos is not None:
                vx = cx - self.last_pos[0]
                vy = cy - self.last_pos[1]
                # Smooth velocity (exponential moving average)
                self.velocity = (0.6 * self.velocity[0] + 0.4 * vx, 0.6 * self.velocity[1] + 0.4 * vy)
            else:
                self.velocity = (0.0, 0.0)

            self.last_pos = (cx, cy)
            self.last_box = box
            self.missing_count = 0

            return TrackedEntity(
                track_id=0,
                bbox=np.array(box, dtype=np.float32),
                class_name="ball",
                confidence=conf,
                frame_idx=frame_idx,
                ground_point=(cx, cy),
                team="Ball",
            )

        # Extrapolation / Dead Reckoning when ball is occluded for short bursts
        if self.last_pos is not None and self.missing_count < self.max_predict_frames:
            self.missing_count += 1
            cx = self.last_pos[0] + self.velocity[0] * 0.85  # decay
            cy = self.last_pos[1] + self.velocity[1] * 0.85
            self.last_pos = (cx, cy)
            self.velocity = (self.velocity[0] * 0.85, self.velocity[1] * 0.85)

            # Reconstruct estimated box (approx 16x16 px)
            box = (cx - 8.0, cy - 8.0, cx + 8.0, cy + 8.0)
            return TrackedEntity(
                track_id=0,
                bbox=np.array(box, dtype=np.float32),
                class_name="ball",
                confidence=0.45,
                frame_idx=frame_idx,
                ground_point=(cx, cy),
                team="Ball",
            )

        # Ball lost
        self.missing_count += 1
        return None
