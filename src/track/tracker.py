"""ByteTrack Multi-Object Tracking Integration for Football Players, Referees, and Ball.

Maintains persistent track IDs across frames, recovers tracks after occlusions (<1s),
and calculates ID-switch metrics as required by PRD Section 2.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import supervision as sv

from src.detect.detector import DetectionResult


@dataclass
class TrackedEntity:
    """Representation of an active or historical tracked entity."""
    track_id: int
    bbox: np.ndarray       # [x1, y1, x2, y2]
    class_name: str        # 'player', 'referee', 'goalkeeper', 'ball'
    confidence: float
    frame_idx: int
    ground_point: Optional[Tuple[float, float]] = None  # (x, y) pixels or meters
    team: Optional[str] = None                         # 'Team A', 'Team B', 'Goalkeeper', 'Referee'
    pitch_x: Optional[float] = None                    # Pitch X in meters
    pitch_y: Optional[float] = None                    # Pitch Y in meters
    speed_kmh: float = 0.0


@dataclass
class FrameTrackingResult:
    """Tracks present in a single frame."""
    frame_idx: int
    players: List[TrackedEntity] = field(default_factory=list)
    ball: Optional[TrackedEntity] = None
    referees: List[TrackedEntity] = field(default_factory=list)

    @property
    def all_entities(self) -> List[TrackedEntity]:
        res = list(self.players) + list(self.referees)
        if self.ball:
            res.append(self.ball)
        return res


class FootballTracker:
    """ByteTrack-based tracker for players and referees with dedicated ball tracker."""

    def __init__(
        self,
        fps: int = 30,
        lost_track_buffer_seconds: float = 1.0,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.8,
    ):
        """Initialize Football Tracker.

        Args:
            fps: Video frames per second.
            lost_track_buffer_seconds: Buffer duration to recover occluded tracks (<1s per PRD).
            track_activation_threshold: Minimum detection confidence to activate a track.
            minimum_matching_threshold: Matching threshold for ByteTrack.
        """
        self.fps = fps
        # PRD specifies recovering IDs after short occlusions (< 1s)
        self.lost_track_buffer = int(fps * lost_track_buffer_seconds)

        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=fps,
        )

        # Track history for metrics and smoothing: track_id -> list of (frame_idx, bbox)
        self.track_history: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        # Ball trajectory history: list of (frame_idx, center_xy, bbox)
        self.ball_history: List[Tuple[int, Tuple[float, float], np.ndarray]] = []

        # ID switch detection
        self.total_unique_tracks: set = set()
        self.id_switches_count: int = 0
        self._last_frame_tracks: Dict[int, np.ndarray] = {}

    def update(self, detection: DetectionResult, frame_idx: int) -> FrameTrackingResult:
        """Update tracker with frame detections.

        Args:
            detection: DetectionResult from FootballDetector.
            frame_idx: Current frame index.

        Returns:
            FrameTrackingResult containing active player tracks, referees, and ball.
        """
        result = FrameTrackingResult(frame_idx=frame_idx)

        # 1. Filter out ball from ByteTrack (ByteTrack tracks human players/refs)
        human_indices = [
            i for i, name in enumerate(detection.class_names)
            if name in ("player", "goalkeeper", "referee")
        ]

        current_frame_tracks: Dict[int, np.ndarray] = {}

        if human_indices and len(detection.boxes) > 0:
            human_boxes = detection.boxes[human_indices]
            human_confs = detection.confidences[human_indices]
            human_classes = detection.class_ids[human_indices]

            sv_detections = sv.Detections(
                xyxy=human_boxes,
                confidence=human_confs,
                class_id=human_classes,
            )

            # Update ByteTrack
            tracked_sv = self.byte_tracker.update_with_detections(sv_detections)

            if tracked_sv.tracker_id is not None:
                for xyxy, conf, cls_id, track_id in zip(
                    tracked_sv.xyxy,
                    tracked_sv.confidence if tracked_sv.confidence is not None else [1.0] * len(tracked_sv),
                    tracked_sv.class_id,
                    tracked_sv.tracker_id,
                ):
                    track_id = int(track_id)
                    self.total_unique_tracks.add(track_id)
                    current_frame_tracks[track_id] = xyxy

                    # Record history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append((frame_idx, xyxy))

                    # Ground contact point (feet: bottom-center)
                    feet_x = float((xyxy[0] + xyxy[2]) / 2.0)
                    feet_y = float(xyxy[3])

                    entity = TrackedEntity(
                        track_id=track_id,
                        bbox=xyxy,
                        class_name="player",  # default, will be refined by classifier
                        confidence=float(conf),
                        frame_idx=frame_idx,
                        ground_point=(feet_x, feet_y),
                    )

                    result.players.append(entity)

        # 2. Track Ball separately (using high-confidence detection + spatial continuity)
        if detection.ball_index is not None and detection.ball_index < len(detection.boxes):
            ball_bbox = detection.boxes[detection.ball_index]
            ball_conf = float(detection.confidences[detection.ball_index])
            bx = float((ball_bbox[0] + ball_bbox[2]) / 2.0)
            by = float((ball_bbox[1] + ball_bbox[3]) / 2.0)

            self.ball_history.append((frame_idx, (bx, by), ball_bbox))
            result.ball = TrackedEntity(
                track_id=0,  # Single ball ID
                bbox=ball_bbox,
                class_name="ball",
                confidence=ball_conf,
                frame_idx=frame_idx,
                ground_point=(bx, by),
            )
        elif self.ball_history:
            # Ball lost in current frame: carry forward with velocity decay if lost < 5 frames
            last_frame, last_pt, last_bbox = self.ball_history[-1]
            if (frame_idx - last_frame) <= 3:
                # Brief occlusion recovery
                result.ball = TrackedEntity(
                    track_id=0,
                    bbox=last_bbox,
                    class_name="ball",
                    confidence=0.5,
                    frame_idx=frame_idx,
                    ground_point=last_pt,
                )

        # 3. Detect ID-switch heuristic (rapid jump in position or overlapping track reassignment)
        self._check_id_switches(current_frame_tracks)
        self._last_frame_tracks = current_frame_tracks

        return result

    def _check_id_switches(self, current_tracks: Dict[int, np.ndarray]):
        """Estimate ID switches based on spatial discontinuity between consecutive frames."""
        for tid, box in current_tracks.items():
            if tid in self._last_frame_tracks:
                prev_box = self._last_frame_tracks[tid]
                c_prev = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2])
                c_curr = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                # If player jumped > 150 pixels in 1 frame (unrealistic for 30fps), likely ID switch or swap
                dist = np.linalg.norm(c_curr - c_prev)
                if dist > 150.0:
                    self.id_switches_count += 1

    def compute_id_switch_rate(self) -> float:
        """Compute ID switches as percentage of total unique tracks (PRD target: < 5%)."""
        total = len(self.total_unique_tracks)
        if total == 0:
            return 0.0
        return (self.id_switches_count / total) * 100.0
