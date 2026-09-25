"""YOLOv11 Object Detection Wrapper for Football Players, Referees, and Ball.

Supports both standard pretrained YOLOv11 models (COCO mapping) and fine-tuned
football dataset models (custom class dictionaries).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class DetectionResult:
    """Standardized detection output for a single frame."""
    boxes: np.ndarray          # Shape (N, 4) in [x1, y1, x2, y2]
    confidences: np.ndarray    # Shape (N,)
    class_ids: np.ndarray      # Shape (N,)
    class_names: List[str]     # Length N list of string labels
    frame_idx: int = 0
    # Filtered subsets for quick access
    player_indices: List[int] = field(default_factory=list)
    ball_index: Optional[int] = None
    referee_indices: List[int] = field(default_factory=list)


class FootballDetector:
    """YOLOv11 Inference Wrapper for Football Computer Vision."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        player_conf: float = 0.30,
        ball_conf: float = 0.15,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """Initialize the football detector.

        Args:
            model_path: Path to YOLOv11 weights (e.g., yolo11n.pt, yolo11s.pt, or football-finetuned.pt)
            player_conf: Confidence threshold for players/referees
            ball_conf: Confidence threshold for ball (lower because ball is small and fast)
            iou_threshold: NMS IoU threshold
            device: 'cuda', 'cuda:0', 'cpu', or None for auto-detection
        """
        import torch
        from ultralytics import YOLO

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path)
        self.player_conf = player_conf
        self.ball_conf = ball_conf
        self.iou_threshold = iou_threshold

        # Inspect model class names
        self.raw_names = self.model.names if hasattr(self.model, "names") else {}
        self.is_custom_football_model = self._check_custom_football_model()

    def _check_custom_football_model(self) -> bool:
        """Determine if loaded model has dedicated football classes or COCO classes."""
        names_lower = [str(name).lower() for name in self.raw_names.values()]
        football_terms = {"player", "ball", "referee", "goalkeeper"}
        return any(term in names_lower for term in football_terms)

    def detect_frame(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """Run detection on a single BGR frame.

        Args:
            frame: BGR numpy image array.
            frame_idx: Frame index for metadata.

        Returns:
            DetectionResult containing boxes, confidences, and labels.
        """
        results = self.model.predict(
            source=frame,
            conf=min(self.player_conf, self.ball_conf),
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        boxes_list: List[List[float]] = []
        conf_list: List[float] = []
        cls_id_list: List[int] = []
        cls_name_list: List[str] = []
        player_indices: List[int] = []
        referee_indices: List[int] = []
        ball_candidates: List[Tuple[int, float, float]] = []  # (index, conf, area)

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            current_idx = 0
            for box, conf, cls_id in zip(boxes, confs, classes):
                raw_name = str(self.raw_names.get(cls_id, "")).lower()

                # Determine role & confidence threshold
                if self.is_custom_football_model:
                    if "ball" in raw_name:
                        if conf < self.ball_conf:
                            continue
                        label = "ball"
                    elif "ref" in raw_name:
                        if conf < self.player_conf:
                            continue
                        label = "referee"
                    elif "goal" in raw_name or "gk" in raw_name:
                        if conf < self.player_conf:
                            continue
                        label = "goalkeeper"
                    elif "player" in raw_name:
                        if conf < self.player_conf:
                            continue
                        label = "player"
                    else:
                        continue
                else:
                    # Standard COCO: class 0 is 'person', class 32 is 'sports ball'
                    if cls_id == 0 or raw_name == "person":
                        if conf < self.player_conf:
                            continue
                        label = "player"
                    elif cls_id == 32 or "ball" in raw_name:
                        if conf < self.ball_conf:
                            continue
                        # Filter out unreasonably large balls (must be < 40x40 typically)
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        if w > 80 or h > 80 or w < 3 or h < 3:
                            continue
                        label = "ball"
                    else:
                        continue

                boxes_list.append(box.tolist())
                conf_list.append(float(conf))
                cls_id_list.append(cls_id)
                cls_name_list.append(label)

                if label in ("player", "goalkeeper"):
                    player_indices.append(current_idx)
                elif label == "referee":
                    referee_indices.append(current_idx)
                elif label == "ball":
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    ball_candidates.append((current_idx, float(conf), float(area)))

                current_idx += 1

        # Select best ball candidate (highest confidence, reasonable size)
        best_ball_idx = None
        if ball_candidates:
            ball_candidates.sort(key=lambda x: x[1], reverse=True)
            best_ball_idx = ball_candidates[0][0]

        return DetectionResult(
            boxes=np.array(boxes_list, dtype=np.float32) if boxes_list else np.empty((0, 4), dtype=np.float32),
            confidences=np.array(conf_list, dtype=np.float32) if conf_list else np.empty((0,), dtype=np.float32),
            class_ids=np.array(cls_id_list, dtype=np.int32) if cls_id_list else np.empty((0,), dtype=np.int32),
            class_names=cls_name_list,
            frame_idx=frame_idx,
            player_indices=player_indices,
            ball_index=best_ball_idx,
            referee_indices=referee_indices,
        )
