"""Jersey Color Clustering and Team Classification.

Extracts dominant jersey colors using K-Means in HSV/Lab space,
clusters tracks into Team A, Team B, Goalkeepers, and Referees,
and persists persistent labels per track across the match.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Classifies players into teams based on dominant jersey color clustering."""

    def __init__(
        self,
        team_a_name: str = "Team A",
        team_b_name: str = "Team B",
        manual_team_a_color: Optional[Tuple[int, int, int]] = None,  # RGB
        manual_team_b_color: Optional[Tuple[int, int, int]] = None,  # RGB
    ):
        """Initialize team classifier.

        Args:
            team_a_name: Display name for first team (e.g., 'VfB Stuttgart' or 'Borussia Mgladbach').
            team_b_name: Display name for second team (e.g., 'Greuther Furth' or 'VfL Wolfsburg').
            manual_team_a_color: Optional reference RGB color for Team A.
            manual_team_b_color: Optional reference RGB color for Team B.
        """
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
        self.manual_team_a_color = manual_team_a_color
        self.manual_team_b_color = manual_team_b_color

        # Store collected color samples per track: track_id -> list of HSV colors
        self.track_color_samples: Dict[int, List[np.ndarray]] = {}
        # Final assigned team labels: track_id -> label string
        self.track_team_labels: Dict[int, str] = {}
        # Representative RGB colors for teams: team_name -> (R, G, B)
        self.team_colors: Dict[str, Tuple[int, int, int]] = {
            self.team_a_name: (220, 20, 60),      # Crimson red default
            self.team_b_name: (30, 144, 255),     # Dodger blue default
            "Goalkeeper": (255, 215, 0),          # Gold default
            "Referee": (50, 205, 50),             # Lime green default
        }

    @staticmethod
    def crop_jersey(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract the upper-body torso (jersey area) from a player bounding box.

        Focuses on y in [15%, 50%] and x in [20%, 80%] of box to exclude
        head/hair, shorts, socks, and grass background edges.
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        bw = x2 - x1
        bh = y2 - y1

        if bw < 10 or bh < 20:
            return None

        # Torso / chest crop
        jy1 = y1 + int(bh * 0.15)
        jy2 = y1 + int(bh * 0.50)
        jx1 = x1 + int(bw * 0.20)
        jx2 = x2 - int(bw * 0.20)

        if jy2 <= jy1 or jx2 <= jx1:
            return None

        jersey_crop = frame[jy1:jy2, jx1:jx2]
        return jersey_crop

    @staticmethod
    def extract_dominant_color_hsv(jersey_crop: np.ndarray, k: int = 2) -> Optional[np.ndarray]:
        """Extract dominant HSV color from jersey crop, filtering out pitch green bleed.

        Returns:
            HSV color vector np.array([H, S, V]) in OpenCV ranges [0-179, 0-255, 0-255].
        """
        if jersey_crop is None or jersey_crop.size == 0:
            return None

        hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        # Filter pitch green pixels: H in [35, 85], S > 40, V > 30
        is_grass = (pixels[:, 0] >= 35) & (pixels[:, 0] <= 85) & (pixels[:, 1] >= 40)
        valid_pixels = pixels[~is_grass]

        # If too few valid non-grass pixels remain, keep all pixels
        if len(valid_pixels) < 15:
            valid_pixels = pixels

        if len(valid_pixels) < k:
            return np.mean(valid_pixels, axis=0) if len(valid_pixels) > 0 else None

        # Run K-Means to find dominant color
        kmeans = KMeans(n_clusters=k, n_init=3, random_state=42)
        kmeans.fit(valid_pixels)

        # Find cluster with largest number of pixels
        counts = np.bincount(kmeans.labels_)
        dominant_idx = np.argmax(counts)
        dominant_hsv = kmeans.cluster_centers_[dominant_idx]
        return dominant_hsv

    def collect_track_color(self, frame: np.ndarray, track_id: int, bbox: np.ndarray):
        """Extract and store a jersey color sample for a track."""
        crop = self.crop_jersey(frame, bbox)
        if crop is not None:
            color = self.extract_dominant_color_hsv(crop)
            if color is not None:
                if track_id not in self.track_color_samples:
                    self.track_color_samples[track_id] = []
                self.track_color_samples[track_id].append(color)

    def fit_team_clusters(self, min_samples_per_track: int = 3):
        """Cluster all accumulated track colors into Team A, Team B, GK, and Referee.

        Persists the assigned team per track.
        """
        track_ids = []
        track_avg_colors = []

        for tid, samples in self.track_color_samples.items():
            if len(samples) >= min_samples_per_track:
                # Use median color across samples to filter out outliers/shadows
                median_color = np.median(samples, axis=0)
                track_ids.append(tid)
                track_avg_colors.append(median_color)

        if not track_ids:
            return

        track_avg_colors = np.array(track_avg_colors)

        # If manual team colors provided: assign to nearest seed
        if self.manual_team_a_color is not None and self.manual_team_b_color is not None:
            a_bgr = np.uint8([[list(reversed(self.manual_team_a_color))]])
            b_bgr = np.uint8([[list(reversed(self.manual_team_b_color))]])
            a_hsv = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)[0, 0]
            b_hsv = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)[0, 0]

            for tid, color in zip(track_ids, track_avg_colors):
                dist_a = np.linalg.norm(color - a_hsv)
                dist_b = np.linalg.norm(color - b_hsv)
                self.track_team_labels[tid] = self.team_a_name if dist_a <= dist_b else self.team_b_name
            return

        # Otherwise: 2-team K-Means clustering on Hue and Saturation
        if len(track_ids) >= 2:
            n_clusters = 2
            # Weight Hue and Value appropriately (circular hue consideration)
            h_rad = (track_avg_colors[:, 0] / 180.0) * 2 * np.pi
            feature_vecs = np.column_stack([
                np.cos(h_rad) * (track_avg_colors[:, 1] / 255.0),
                np.sin(h_rad) * (track_avg_colors[:, 1] / 255.0),
                track_avg_colors[:, 2] / 255.0,
            ])

            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_vecs)

            for tid, label in zip(track_ids, cluster_labels):
                self.track_team_labels[tid] = self.team_a_name if label == 0 else self.team_b_name

            # Derive average team BGR colors for visualization
            for cluster_idx, team_name in [(0, self.team_a_name), (1, self.team_b_name)]:
                members = track_avg_colors[cluster_labels == cluster_idx]
                if len(members) > 0:
                    mean_hsv = np.uint8([[np.mean(members, axis=0)]])
                    bgr = cv2.cvtColor(mean_hsv, cv2.COLOR_HSV2BGR)[0, 0]
                    self.team_colors[team_name] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    def get_track_team(self, track_id: int) -> str:
        """Get persistent team label for track ID, defaulting to team_a_name if unassigned."""
        return self.track_team_labels.get(track_id, self.team_a_name)

    def get_team_color(self, team: str) -> Tuple[int, int, int]:
        """Get RGB color tuple for team label."""
        return self.team_colors.get(team, (200, 200, 200))
