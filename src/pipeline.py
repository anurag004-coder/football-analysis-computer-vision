"""End-to-End Football Analytics Computer Vision Pipeline.

Orchestrates:
1. YOLOv11 detection of players, referees, and ball
2. ByteTrack persistent multi-object tracking and occlusion recovery
3. K-Means jersey color clustering and persistent team classification
4. Pitch homography mapping from camera pixels to 2D pitch metric coordinates (meters)
5. Physical movement metrics (speed, distance)
6. Voronoi space control and tactical pitch dominance
7. Possession state machine and passing network extraction
8. Video rendering: Annotated broadcast video and 2D tactical pitch radar video
9. Export to CSV, Parquet, and JSON analytics reports
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.detect.detector import FootballDetector
from src.track.tracker import FootballTracker, FrameTrackingResult
from src.classify.classifier import TeamClassifier
from src.pitch.pitch_model import PitchModel
from src.pitch.homography import PitchHomography
from src.pitch.calibrator import PitchCalibrator
from src.analytics.metrics import MovementAnalytics
from src.analytics.voronoi import VoronoiSpaceControl
from src.analytics.passing import PassingNetworkAnalytics
from src.analytics.visualizer import TacticalVisualizer
from src.track.ball_tracker import FootballBallTracker


class FootballAnalysisPipeline:
    """Master pipeline orchestrator for football video analysis."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: Optional[str] = None,
        player_conf: float = 0.30,
        ball_conf: float = 0.15,
        calibration_file: Optional[str] = None,
        team_a_name: str = "Team A",
        team_b_name: str = "Team B",
        manual_team_a_color: Optional[str] = None,  # Hex string '#RRGGBB'
        manual_team_b_color: Optional[str] = None,
    ):
        print(f"[Pipeline] Initializing Football Analytics Pipeline: {team_a_name} vs {team_b_name}...")
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
        self.detector = FootballDetector(
            model_path=model_path,
            player_conf=player_conf,
            ball_conf=ball_conf,
            device=device,
        )

        self.calibration_file = calibration_file
        self.homography: Optional[PitchHomography] = None

        # Parse manual colors if provided
        self.team_a_rgb = self._parse_hex_color(manual_team_a_color) if manual_team_a_color else None
        self.team_b_rgb = self._parse_hex_color(manual_team_b_color) if manual_team_b_color else None
        self.classifier = TeamClassifier(
            team_a_name=self.team_a_name,
            team_b_name=self.team_b_name,
            manual_team_a_color=self.team_a_rgb,
            manual_team_b_color=self.team_b_rgb,
        )

        self.pitch_model = PitchModel()
        self.calibrator = PitchCalibrator(self.pitch_model)

    @staticmethod
    def transcode_to_h264_web(input_path: Path, output_path: Path, snapshot_frame: int = 40):
        """Transcode video to web-compatible H.264 MP4 using PyAV and save snapshot."""
        try:
            import av
            cap = cv2.VideoCapture(str(input_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                cap.release()
                return

            out_container = av.open(str(output_path), mode="w")
            stream = out_container.add_stream("h264", rate=int(fps))
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "22", "preset": "veryfast"}

            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx == snapshot_frame:
                    snap_path = output_path.parent / f"{output_path.stem}_snapshot.jpg"
                    cv2.imwrite(str(snap_path), frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(av_frame):
                    out_container.mux(packet)
                idx += 1

            for packet in stream.encode():
                out_container.mux(packet)
            out_container.close()
            cap.release()
        except Exception as e:
            print(f"[Pipeline] Web transcode notice: {e}")

    @staticmethod
    def _parse_hex_color(hex_str: str) -> Optional[tuple]:
        hex_clean = hex_str.lstrip("#")
        if len(hex_clean) == 6:
            return tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        return None

    @staticmethod
    def _rgb_to_hex(rgb: Optional[tuple]) -> str:
        if rgb and len(rgb) == 3:
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        return "#FFFFFF"

    def process_video(
        self,
        video_path: str,
        output_dir: str = "outputs",
        max_frames: Optional[int] = None,
        render_annotated: bool = True,
        render_tactical_map: bool = True,
        render_side_by_side: bool = True,
        interactive_calibration: bool = False,
    ) -> Dict:
        """Run the complete pipeline on an input video clip.

        Args:
            video_path: Path to input MP4 video.
            output_dir: Destination directory for all artifacts.
            max_frames: Optional maximum number of frames to process.
            render_annotated: Whether to generate annotated broadcast video.
            render_tactical_map: Whether to generate 2D pitch radar map video.
            render_side_by_side: Whether to generate combined side-by-side video.
            interactive_calibration: Launch interactive window to click pitch landmarks.

        Returns:
            Dictionary with processing summary, metrics, and artifact file paths.
        """
        start_time = time.time()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
        print(f"[Pipeline] Video: {video_path} ({width}x{height} @ {fps:.1f} fps, {frames_to_process} frames)")

        # 1. Initialize Trackers and Analytics Engines
        tracker = FootballTracker(fps=int(fps))
        ball_tracker = FootballBallTracker()
        movement_analytics = MovementAnalytics(fps=fps)
        voronoi_engine = VoronoiSpaceControl()
        team_a_hex = self._rgb_to_hex(self.team_a_rgb) if self.team_a_rgb else "#DC143C"
        team_b_hex = self._rgb_to_hex(self.team_b_rgb) if self.team_b_rgb else "#FFFFFF"
        passing_engine = PassingNetworkAnalytics(
            team_a_name=self.team_a_name,
            team_b_name=self.team_b_name,
            team_a_color=team_a_hex,
            team_b_color=team_b_hex,
        )
        # Convert RGB to BGR for OpenCV visualizer if manual colors specified
        team_a_bgr = tuple(reversed(self.team_a_rgb)) if self.team_a_rgb else None
        team_b_bgr = tuple(reversed(self.team_b_rgb)) if self.team_b_rgb else None
        visualizer = TacticalVisualizer(
            team_a_name=self.team_a_name,
            team_b_name=self.team_b_name,
            team_a_bgr=team_a_bgr,
            team_b_bgr=team_b_bgr,
        )

        # 2. Setup Pitch Homography
        if self.calibration_file and Path(self.calibration_file).exists():
            print(f"[Pipeline] Loading calibration from {self.calibration_file}")
            self.homography = PitchHomography.load_calibration(self.calibration_file)
        elif interactive_calibration:
            ret, first_frame = cap.read()
            if ret:
                print("[Pipeline] Launching interactive pitch calibration...")
                self.homography, reproj_err, _ = self.calibrator.interactive_calibrate(first_frame)
                print(f"[Pipeline] Calibration complete. Reprojection error: {reproj_err:.2f} px")
                calib_save = out_path / "calibration.json"
                self.homography.save_calibration(calib_save, {"reprojection_error": reproj_err})
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("[Pipeline] Using preset broadcast pitch calibration...")
            self.homography = self.calibrator.create_broadcast_preset(width, height)

        # 3. Setup Video Writers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = None
        tactical_writer = None
        combined_writer = None

        stem = Path(video_path).stem
        annotated_video_path = out_path / f"{stem}_annotated.mp4"
        tactical_video_path = out_path / f"{stem}_tactical_map.mp4"
        combined_video_path = out_path / f"{stem}_combined.mp4"

        if render_annotated:
            annotated_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))
        if render_tactical_map:
            tactical_writer = cv2.VideoWriter(
                str(tactical_video_path), fourcc, fps, (visualizer.radar_width, visualizer.radar_height)
            )
        if render_side_by_side:
            combined_w = width + int(visualizer.radar_width * (height / visualizer.radar_height))
            combined_writer = cv2.VideoWriter(str(combined_video_path), fourcc, fps, (combined_w, height))

        # Data collection containers for structured tracking data (CSV / Parquet)
        tracking_records: List[Dict] = []
        frame_tracking_results: List[FrameTrackingResult] = []

        print("[Pipeline] Pass 1: Detection, Tracking, and Jersey Sampling...")
        frame_idx = 0
        pbar = tqdm(total=frames_to_process, desc="Tracking & Classifying")

        while cap.isOpened() and frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # A. Detection
            detections = self.detector.detect_frame(frame, frame_idx=frame_idx)

            # B. Tracking
            tracking_res = tracker.update(detections, frame_idx=frame_idx)

            # B2. Ball Tracking & Proximity Association
            player_boxes = [p.bbox for p in tracking_res.players]
            yolo_ball_box = detections.boxes[detections.ball_index] if detections.ball_index is not None and len(detections.boxes) > 0 else None
            tracked_ball = ball_tracker.update(frame, yolo_ball_box, player_boxes, frame_idx=frame_idx)
            if tracked_ball is not None:
                tracking_res.ball = tracked_ball

            # C. Jersey Color Sample Accumulation for Players
            for player in tracking_res.players:
                self.classifier.collect_track_color(frame, player.track_id, player.bbox)

            frame_tracking_results.append(tracking_res)
            frame_idx += 1
            pbar.update(1)

        pbar.close()

        # D. Global Team Clustering: Fit K-Means on accumulated jersey colors
        print("[Pipeline] Clustering jersey colors into teams...")
        self.classifier.fit_team_clusters()

        # 4. Pass 2: Homography Projection, Analytics Computation, and Video Rendering
        print("[Pipeline] Pass 2: Pitch Mapping, Tactical Analytics & Video Rendering...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pbar = tqdm(total=len(frame_tracking_results), desc="Rendering Analytics")

        for frame_idx, tracking_res in enumerate(frame_tracking_results):
            ret, frame = cap.read()
            if not ret:
                break

            player_pitch_coords: List[Tuple[float, float]] = []
            player_teams: List[str] = []
            player_tids: List[int] = []
            player_positions_map: Dict[int, Tuple[float, float]] = {}
            player_teams_map: Dict[int, str] = {}

            # Project player positions to pitch metric coordinates (meters)
            for player in tracking_res.players:
                player.team = self.classifier.get_track_team(player.track_id)
                if player.ground_point and self.homography:
                    pitch_xy = self.homography.image_to_pitch([player.ground_point])[0]
                    player.pitch_x = float(pitch_xy[0])
                    player.pitch_y = float(pitch_xy[1])

                    # Record for movement metrics
                    movement_analytics.add_position(player.track_id, frame_idx, player.pitch_x, player.pitch_y)
                    player.speed_kmh = movement_analytics.get_instantaneous_speed(player.track_id, frame_idx)

                    player_pitch_coords.append((player.pitch_x, player.pitch_y))
                    player_teams.append(player.team)
                    player_tids.append(player.track_id)
                    player_positions_map[player.track_id] = (player.pitch_x, player.pitch_y)
                    player_teams_map[player.track_id] = player.team

                # Append to structured tracking records
                tracking_records.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "track_id": player.track_id,
                    "entity_type": "player",
                    "team": player.team,
                    "confidence": round(player.confidence, 3),
                    "bbox_x1": round(float(player.bbox[0]), 1),
                    "bbox_y1": round(float(player.bbox[1]), 1),
                    "bbox_x2": round(float(player.bbox[2]), 1),
                    "bbox_y2": round(float(player.bbox[3]), 1),
                    "pitch_x_m": round(player.pitch_x, 2) if player.pitch_x is not None else None,
                    "pitch_y_m": round(player.pitch_y, 2) if player.pitch_y is not None else None,
                    "speed_kmh": round(player.speed_kmh, 1),
                })

            # Project Ball
            ball_pitch_xy = None
            if tracking_res.ball and tracking_res.ball.ground_point and self.homography:
                b_proj = self.homography.image_to_pitch([tracking_res.ball.ground_point])[0]
                tracking_res.ball.pitch_x = float(b_proj[0])
                tracking_res.ball.pitch_y = float(b_proj[1])
                ball_pitch_xy = (tracking_res.ball.pitch_x, tracking_res.ball.pitch_y)

                tracking_records.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "track_id": 0,
                    "entity_type": "ball",
                    "team": "Ball",
                    "confidence": round(tracking_res.ball.confidence, 3),
                    "bbox_x1": round(float(tracking_res.ball.bbox[0]), 1),
                    "bbox_y1": round(float(tracking_res.ball.bbox[1]), 1),
                    "bbox_x2": round(float(tracking_res.ball.bbox[2]), 1),
                    "bbox_y2": round(float(tracking_res.ball.bbox[3]), 1),
                    "pitch_x_m": round(tracking_res.ball.pitch_x, 2),
                    "pitch_y_m": round(tracking_res.ball.pitch_y, 2),
                    "speed_kmh": 0.0,
                })

            # Compute Voronoi space control
            voronoi_res = None
            if len(player_pitch_coords) >= 3:
                voronoi_res = voronoi_engine.compute_frame_voronoi(
                    np.array(player_pitch_coords), player_teams, player_tids
                )

            # Update possession and passing heuristics
            passing_engine.update_frame(
                frame_idx=frame_idx,
                ball_pos=ball_pitch_xy,
                player_positions=player_positions_map,
                player_teams=player_teams_map,
            )

            # Render Visualizations
            annotated_frame = visualizer.annotate_broadcast_frame(frame, tracking_res)
            radar_frame = visualizer.draw_2d_pitch_radar(
                tracking_result=tracking_res,
                voronoi_data=voronoi_res,
                possession_pct=passing_engine.get_possession_percentages(),
            )

            if annotated_writer:
                annotated_writer.write(annotated_frame)
            if tactical_writer:
                tactical_writer.write(radar_frame)
            if combined_writer:
                combined_frame = visualizer.compose_side_by_side(annotated_frame, radar_frame)
                combined_writer.write(combined_frame)

            pbar.update(1)

        pbar.close()
        cap.release()
        if annotated_writer:
            annotated_writer.release()
        if tactical_writer:
            tactical_writer.release()
        if combined_writer:
            combined_writer.release()

        # Web H.264 Transcoding for browser playback
        web_combined_path = out_path / f"{stem}_combined_web.mp4"
        web_annotated_path = out_path / f"{stem}_annotated_web.mp4"
        web_tactical_path = out_path / f"{stem}_tactical_map_web.mp4"

        if render_side_by_side and combined_video_path.exists():
            self.transcode_to_h264_web(combined_video_path, web_combined_path)
        if render_annotated and annotated_video_path.exists():
            self.transcode_to_h264_web(annotated_video_path, web_annotated_path)
        if render_tactical_map and tactical_video_path.exists():
            self.transcode_to_h264_web(tactical_video_path, web_tactical_path)

        # 5. Export Structured Data (CSV and Parquet)
        print("[Pipeline] Exporting structured tracking data...")
        tracking_df = pd.DataFrame(tracking_records)
        csv_path = out_path / f"{stem}_tracking_data.csv"
        parquet_path = out_path / f"{stem}_tracking_data.parquet"
        tracking_df.to_csv(csv_path, index=False)
        try:
            tracking_df.to_parquet(parquet_path, index=False)
        except Exception as e:
            print(f"[Pipeline] Parquet export note: {e}")

        # 6. Compute Final Summary Metrics & Passing Networks
        print("[Pipeline] Compiling tactical analytics report...")
        movement_df = movement_analytics.compute_all_metrics()
        movement_csv_path = out_path / f"{stem}_player_physical_metrics.csv"
        movement_df.to_csv(movement_csv_path, index=False)

        # Plot Passing Networks
        passing_net_a_path = out_path / f"{stem}_passing_network_{self.team_a_name.lower().replace(' ', '_')}.png"
        passing_net_b_path = out_path / f"{stem}_passing_network_{self.team_b_name.lower().replace(' ', '_')}.png"
        passing_engine.plot_passing_network(self.team_a_name, str(passing_net_a_path))
        passing_engine.plot_passing_network(self.team_b_name, str(passing_net_b_path))

        id_switch_rate = tracker.compute_id_switch_rate()
        elapsed_sec = time.time() - start_time

        summary_report = {
            "input_video": str(video_path),
            "total_frames_processed": frame_idx,
            "fps": fps,
            "processing_time_sec": round(elapsed_sec, 2),
            "id_switch_rate_percent": round(id_switch_rate, 2),
            "total_unique_tracks": len(tracker.total_unique_tracks),
            "id_switches_count": tracker.id_switches_count,
            "possession_percentages": passing_engine.get_possession_percentages(),
            "total_passes_detected": len(passing_engine.pass_events),
            "tactical_metrics": {
                self.team_a_name: passing_engine.get_team_tactical_metrics(self.team_a_name),
                self.team_b_name: passing_engine.get_team_tactical_metrics(self.team_b_name),
            },
            "outputs": {
                "annotated_video": str(annotated_video_path) if render_annotated else None,
                "tactical_map_video": str(tactical_video_path) if render_tactical_map else None,
                "combined_video": str(combined_video_path) if render_side_by_side else None,
                "tracking_csv": str(csv_path),
                "tracking_parquet": str(parquet_path),
                "physical_metrics_csv": str(movement_csv_path),
                "passing_network_team_a": str(passing_net_a_path),
                "passing_network_team_b": str(passing_net_b_path),
            },
        }

        report_json_path = out_path / f"{stem}_analysis_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE")
        print(f"Processed: {frame_idx} frames in {elapsed_sec:.1f}s ({frame_idx / max(elapsed_sec, 0.001):.1f} fps)")
        print(f"ID Switch Rate: {id_switch_rate:.2f}% (PRD Target: < 5.0%)")
        print(f"Annotated Video: {annotated_video_path}")
        print(f"Tactical 2D Map: {tactical_video_path}")
        print(f"Tracking Data: {csv_path}")
        print("=" * 60 + "\n")

        return summary_report
