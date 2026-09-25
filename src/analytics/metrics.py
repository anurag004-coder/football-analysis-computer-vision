"""Physical Performance Metrics: Speed, Distance, and Workload Analytics.

Calculates player movement metrics from calibrated 2D pitch coordinates:
- Instantaneous speed (km/h) with noise filtering
- Total distance covered (meters and km)
- Maximum sprint speed
- Team cumulative distance and average speeds
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class MovementAnalytics:
    """Calculates physical tracking metrics from 2D metric pitch trajectories."""

    def __init__(self, fps: float = 30.0, smoothing_window: int = 5, max_plausible_speed_kmh: float = 38.0):
        """Initialize movement analytics.

        Args:
            fps: Video frame rate.
            smoothing_window: Number of frames for moving average speed smoothing.
            max_plausible_speed_kmh: Upper physical cap for human sprint speed.
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.smoothing_window = smoothing_window
        self.max_plausible_speed = max_plausible_speed_kmh

        # History per track: track_id -> list of (frame_idx, x_meters, y_meters)
        self.trajectories: Dict[int, List[Tuple[int, float, float]]] = {}
        # Computed smoothed speeds: track_id -> dict of frame_idx -> speed_kmh
        self.speed_cache: Dict[int, Dict[int, float]] = {}

    def add_position(self, track_id: int, frame_idx: int, x_meters: float, y_meters: float):
        """Record a player's metric pitch position at a specific frame."""
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
        self.trajectories[track_id].append((frame_idx, float(x_meters), float(y_meters)))

    def compute_all_metrics(self) -> pd.DataFrame:
        """Compute comprehensive performance summary table across all tracks.

        Returns:
            DataFrame with columns: [track_id, total_distance_m, avg_speed_kmh, max_speed_kmh, frames_tracked]
        """
        records = []
        for track_id, pos_list in self.trajectories.items():
            if len(pos_list) < 2:
                continue

            # Sort by frame index
            sorted_pos = sorted(pos_list, key=lambda p: p[0])
            frames = np.array([p[0] for p in sorted_pos])
            coords = np.array([[p[1], p[2]] for p in sorted_pos])

            # Frame differences and distances
            frame_diffs = np.diff(frames)
            # Avoid division by zero
            time_diffs = np.maximum(frame_diffs, 1) * self.dt

            step_distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)

            # Raw speeds (m/s)
            raw_speeds_ms = step_distances / time_diffs
            raw_speeds_kmh = raw_speeds_ms * 3.6

            # Filter unrealistic teleportation spikes (e.g. tracker re-association)
            valid_mask = raw_speeds_kmh <= self.max_plausible_speed
            filtered_distances = step_distances[valid_mask]
            total_dist_m = float(np.sum(filtered_distances))

            # Apply moving average filter to speeds
            if len(raw_speeds_kmh) >= self.smoothing_window:
                smoothed_kmh = np.convolve(
                    raw_speeds_kmh,
                    np.ones(self.smoothing_window) / self.smoothing_window,
                    mode="same",
                )
            else:
                smoothed_kmh = raw_speeds_kmh

            smoothed_kmh = np.clip(smoothed_kmh, 0.0, self.max_plausible_speed)

            # Cache speeds per frame for fast lookup
            if track_id not in self.speed_cache:
                self.speed_cache[track_id] = {}
            for f_idx, spd in zip(frames[1:], smoothed_kmh):
                self.speed_cache[track_id][f_idx] = float(spd)

            avg_spd = float(np.mean(smoothed_kmh)) if len(smoothed_kmh) > 0 else 0.0
            max_spd = float(np.max(smoothed_kmh)) if len(smoothed_kmh) > 0 else 0.0

            records.append({
                "track_id": track_id,
                "total_distance_m": round(total_dist_m, 2),
                "avg_speed_kmh": round(avg_spd, 2),
                "max_speed_kmh": round(max_spd, 2),
                "frames_tracked": len(pos_list),
            })

        df = pd.DataFrame(records)
        return df

    def get_instantaneous_speed(self, track_id: int, frame_idx: int) -> float:
        """Get smoothed instantaneous speed (km/h) for a specific track at frame."""
        if track_id in self.speed_cache and frame_idx in self.speed_cache[track_id]:
            return self.speed_cache[track_id][frame_idx]

        # If not cached yet, compute from recent trajectory window
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 2:
            return 0.0

        hist = self.trajectories[track_id]
        if len(hist) >= 2:
            p_curr = hist[-1]
            p_prev = hist[-2]
            df = p_curr[0] - p_prev[0]
            if 0 < df <= 5:
                dist = np.hypot(p_curr[1] - p_prev[1], p_curr[2] - p_prev[2])
                spd = (dist / (df * self.dt)) * 3.6
                return min(round(float(spd), 1), self.max_plausible_speed)
        return 0.0
