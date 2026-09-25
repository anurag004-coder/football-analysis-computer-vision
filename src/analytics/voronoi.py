"""Voronoi Space Control and Pitch Dominance Analysis.

Computes pitch space partitions and team pitch control percentages
using bounded Voronoi tessellation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import Voronoi


class VoronoiSpaceControl:
    """Computes Voronoi space control polygons and team dominance percentages."""

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.total_pitch_area = pitch_length * pitch_width

    def compute_frame_voronoi(
        self,
        player_points: np.ndarray,      # Shape (N, 2) in meters
        team_labels: List[str],         # Length N ('Team A' or 'Team B')
        track_ids: Optional[List[int]] = None,
    ) -> Dict:
        """Compute bounded Voronoi polygons and space control metrics for a frame.

        Args:
            player_points: Array of (X, Y) coordinates in pitch metric space (meters).
            team_labels: List of team names corresponding to each player.
            track_ids: Optional list of track IDs.

        Returns:
            Dictionary with:
                'team_control_pct': {'Team A': float, 'Team B': float},
                'polygons': list of (track_id, team_label, polygon_vertices),
        """
        result = {
            "team_control_pct": {"Team A": 50.0, "Team B": 50.0},
            "polygons": [],
        }

        N = len(player_points)
        if N < 3:
            return result

        # Filter out players that are outside pitch margins
        valid_indices = []
        for i, (x, y) in enumerate(player_points):
            if -2.0 <= x <= self.pitch_length + 2.0 and -2.0 <= y <= self.pitch_width + 2.0:
                valid_indices.append(i)

        if len(valid_indices) < 3:
            return result

        pts = np.asarray(player_points[valid_indices], dtype=np.float64)
        teams = [team_labels[i] for i in valid_indices]
        tids = [track_ids[i] if track_ids else i for i in valid_indices]

        # Mirror points across boundaries to ensure all pitch cells are bounded
        mirrored_pts = []
        # Left boundary (x = 0)
        mirrored_pts.append(np.column_stack([-pts[:, 0], pts[:, 1]]))
        # Right boundary (x = pitch_length)
        mirrored_pts.append(np.column_stack([2 * self.pitch_length - pts[:, 0], pts[:, 1]]))
        # Top boundary (y = 0)
        mirrored_pts.append(np.column_stack([pts[:, 0], -pts[:, 1]]))
        # Bottom boundary (y = pitch_width)
        mirrored_pts.append(np.column_stack([pts[:, 0], 2 * self.pitch_width - pts[:, 1]]))

        all_points = np.vstack([pts] + mirrored_pts)

        try:
            vor = Voronoi(all_points)
        except Exception:
            return result

        # Extract cells for original N players and clip to pitch rectangle
        team_areas = {t: 0.0 for t in set(teams)}

        for i, (point, team, tid) in enumerate(zip(pts, teams, tids)):
            region_idx = vor.point_region[i]
            region_vertices_idx = vor.regions[region_idx]

            if not region_vertices_idx or -1 in region_vertices_idx:
                continue

            polygon = vor.vertices[region_vertices_idx]
            clipped_poly = self._clip_polygon_to_pitch(polygon)

            if len(clipped_poly) >= 3:
                area = self._polygon_area(clipped_poly)
                if team in team_areas:
                    team_areas[team] += area
                result["polygons"].append({
                    "track_id": tid,
                    "team": team,
                    "centroid": point.tolist(),
                    "polygon": clipped_poly.tolist(),
                    "area_m2": round(area, 1),
                })

        total_area = sum(team_areas.values())
        if total_area > 0:
            result["team_control_pct"] = {
                t: round((area / total_area) * 100.0, 1) for t, area in team_areas.items()
            }

        return result

    def _clip_polygon_to_pitch(self, polygon: np.ndarray) -> np.ndarray:
        """Sutherland-Hodgman polygon clipping against pitch rectangle [0, L] x [0, W]."""
        # Bounding box edges: (line_point, normal pointing inward)
        clip_edges = [
            (np.array([0.0, 0.0]), np.array([1.0, 0.0])),                # Left (x >= 0)
            (np.array([self.pitch_length, 0.0]), np.array([-1.0, 0.0])), # Right (x <= L)
            (np.array([0.0, 0.0]), np.array([0.0, 1.0])),                # Top (y >= 0)
            (np.array([0.0, self.pitch_width]), np.array([0.0, -1.0])),  # Bottom (y <= W)
        ]

        output_poly = polygon
        for edge_pt, normal in clip_edges:
            input_poly = output_poly
            output_poly = []
            if len(input_poly) == 0:
                break

            s = input_poly[-1]
            for p in input_poly:
                # Test if point is inside half-plane
                p_inside = np.dot(p - edge_pt, normal) >= 0
                s_inside = np.dot(s - edge_pt, normal) >= 0

                if p_inside:
                    if not s_inside:
                        output_poly.append(self._line_intersection(s, p, edge_pt, normal))
                    output_poly.append(p)
                elif s_inside:
                    output_poly.append(self._line_intersection(s, p, edge_pt, normal))
                s = p
            output_poly = np.array(output_poly)

        return output_poly

    @staticmethod
    def _line_intersection(p1: np.ndarray, p2: np.ndarray, edge_pt: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Find intersection of line segment (p1 -> p2) with plane through edge_pt with normal."""
        u = p2 - p1
        dot = np.dot(normal, u)
        if abs(dot) < 1e-7:
            return p1
        w = p1 - edge_pt
        fac = -np.dot(normal, w) / dot
        return p1 + fac * u

    @staticmethod
    def _polygon_area(vertices: np.ndarray) -> float:
        """Shoelace formula for polygon area."""
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
