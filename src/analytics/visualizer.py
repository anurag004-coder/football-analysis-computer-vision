"""Tactical Visualizer and Video Annotation Engine.

Renders:
1. Annotated broadcast video (team-colored player halos, track IDs, speeds, ball tracker)
2. 2D top-down tactical radar video (pitch markings, player dots, Voronoi space control polygons)
3. Combined side-by-side analytics video
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.track.tracker import FrameTrackingResult, TrackedEntity
from src.pitch.pitch_model import PitchDimensions


class TacticalVisualizer:
    """Renders professional tactical overlays on broadcast frames and creates 2D pitch radar maps."""

    def __init__(
        self,
        pitch_dim: Optional[PitchDimensions] = None,
        radar_width: int = 640,
        radar_height: int = 420,
        team_a_name: str = "Team A",
        team_b_name: str = "Team B",
        team_a_bgr: Optional[Tuple[int, int, int]] = None,
        team_b_bgr: Optional[Tuple[int, int, int]] = None,
    ):
        self.pitch_dim = pitch_dim or PitchDimensions()
        self.radar_width = radar_width
        self.radar_height = radar_height
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name

        # Tactical Pitch Margins inside radar canvas
        self.margin = 25
        self.field_w = self.radar_width - 2 * self.margin
        self.field_h = self.radar_height - 2 * self.margin

        # Team display colors (BGR)
        self.bgr_colors = {
            self.team_a_name: team_a_bgr or (60, 20, 220),       # Crimson Red default
            self.team_b_name: team_b_bgr or (255, 144, 30),      # Dodger Blue default
            "Team A": (60, 20, 220),
            "Team B": (255, 144, 30),
            "Goalkeeper": (0, 215, 255),   # Gold
            "Referee": (50, 205, 50),      # Lime Green
            "Ball": (0, 255, 255),         # Yellow
        }

    def pitch_to_radar_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert pitch metric coordinates (meters) to 2D radar pixel coordinates."""
        px = int(self.margin + (x_m / self.pitch_dim.length) * self.field_w)
        py = int(self.margin + (y_m / self.pitch_dim.width) * self.field_h)
        return px, py

    def draw_2d_pitch_radar(
        self,
        tracking_result: FrameTrackingResult,
        voronoi_data: Optional[Dict] = None,
        possession_pct: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Render standalone 2D top-down tactical radar map for a frame.

        Args:
            tracking_result: Current frame tracking result.
            voronoi_data: Output from VoronoiSpaceControl.
            possession_pct: Dict of Team A and Team B possession percentages.

        Returns:
            BGR image array of the 2D pitch radar.
        """
        canvas = np.full((self.radar_height, self.radar_width, 3), (25, 45, 25), dtype=np.uint8)

        # 1. Draw Voronoi Space Control Polygons (semi-transparent)
        if voronoi_data and "polygons" in voronoi_data and voronoi_data["polygons"]:
            overlay = canvas.copy()
            for poly_info in voronoi_data["polygons"]:
                team = poly_info["team"]
                pts_m = poly_info["polygon"]
                pixel_pts = np.array([self.pitch_to_radar_pixel(p[0], p[1]) for p in pts_m], dtype=np.int32)
                color = self.bgr_colors.get(team, (100, 100, 100))
                cv2.fillPoly(overlay, [pixel_pts], color)
            # Blend overlay with 30% opacity
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        # 2. Draw Pitch Markings (White Lines)
        line_color = (240, 240, 240)
        thickness = 1

        # Outer boundary
        tl = (self.margin, self.margin)
        br = (self.margin + self.field_w, self.margin + self.field_h)
        cv2.rectangle(canvas, tl, br, line_color, thickness)

        # Halfway line
        half_x = self.margin + self.field_w // 2
        cv2.line(canvas, (half_x, self.margin), (half_x, self.margin + self.field_h), line_color, thickness)

        # Center circle & spot
        center_px, center_py = self.pitch_to_radar_pixel(self.pitch_dim.length / 2, self.pitch_dim.width / 2)
        radius_px = int((self.pitch_dim.center_circle_radius / self.pitch_dim.length) * self.field_w)
        cv2.circle(canvas, (center_px, center_py), radius_px, line_color, thickness)
        cv2.circle(canvas, (center_px, center_py), 3, line_color, -1)

        # Penalty boxes
        # Left box
        p_len_px = int((self.pitch_dim.penalty_box_length / self.pitch_dim.length) * self.field_w)
        p_top_py, _ = self.pitch_to_radar_pixel(0, (self.pitch_dim.width - self.pitch_dim.penalty_box_width) / 2)
        p_bot_py, _ = self.pitch_to_radar_pixel(0, (self.pitch_dim.width + self.pitch_dim.penalty_box_width) / 2)
        cv2.rectangle(canvas, (self.margin, p_top_py), (self.margin + p_len_px, p_bot_py), line_color, thickness)

        # Right box
        r_box_left = self.margin + self.field_w - p_len_px
        cv2.rectangle(canvas, (r_box_left, p_top_py), (self.margin + self.field_w, p_bot_py), line_color, thickness)

        # 3. Draw Players as Tactical Dots
        for p in tracking_result.players:
            if p.pitch_x is not None and p.pitch_y is not None:
                px, py = self.pitch_to_radar_pixel(p.pitch_x, p.pitch_y)
                if 0 <= px < self.radar_width and 0 <= py < self.radar_height:
                    team_col = self.bgr_colors.get(p.team, (200, 200, 200))
                    # Outer black rim for contrast
                    cv2.circle(canvas, (px, py), 8, (0, 0, 0), -1)
                    # Filled team dot
                    cv2.circle(canvas, (px, py), 6, team_col, -1)
                    
                    # ID label: high-contrast text based on team color luminance
                    lum = 0.299 * team_col[2] + 0.587 * team_col[1] + 0.114 * team_col[0]
                    text_col = (15, 15, 15) if lum > 150 else (255, 255, 255)
                    id_str = str(p.track_id)
                    font_scale = 0.26 if len(id_str) > 2 else 0.30
                    cv2.putText(canvas, id_str, (px - 5, py + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, 1, cv2.LINE_AA)

        # 4. Draw Ball
        if tracking_result.ball and tracking_result.ball.pitch_x is not None:
            bx, by = self.pitch_to_radar_pixel(tracking_result.ball.pitch_x, tracking_result.ball.pitch_y)
            cv2.circle(canvas, (bx, by), 5, (0, 255, 255), -1)
            cv2.circle(canvas, (bx, by), 6, (0, 0, 0), 1)

        # 5. Header HUD: Space Control & Possession
        header_y = 16
        if voronoi_data and "team_control_pct" in voronoi_data:
            pct_a = voronoi_data["team_control_pct"].get(self.team_a_name, 50.0)
            pct_b = voronoi_data["team_control_pct"].get(self.team_b_name, 50.0)
            hud_text = f"Space: {self.team_a_name} {pct_a}% | {pct_b}% {self.team_b_name}"
            cv2.putText(canvas, hud_text, (self.margin, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        return canvas

    def annotate_broadcast_frame(
        self,
        frame: np.ndarray,
        tracking_result: FrameTrackingResult,
    ) -> np.ndarray:
        """Annotate broadcast video frame with bounding boxes, halos, track IDs, and speeds.

        Args:
            frame: Original BGR video frame.
            tracking_result: Tracking result for current frame.

        Returns:
            Annotated BGR video frame.
        """
        annotated = frame.copy()
        frame_h, frame_w = annotated.shape[:2]

        # 1. Annotate Players
        for p in tracking_result.players:
            x1, y1, x2, y2 = map(int, p.bbox)
            team_col = self.bgr_colors.get(p.team, (200, 200, 200))

            # Ground contact ellipse (feet halo)
            center_x = (x1 + x2) // 2
            axes = ((x2 - x1) // 2, max(6, (x2 - x1) // 5))
            cv2.ellipse(annotated, (center_x, y2), axes, 0, 0, 360, team_col, 2)

            # Subtle bounding box with dark shadow
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), team_col, 1)

            # High-contrast Broadcast Badge above player
            spd_text = f"{p.speed_kmh:.1f} km/h" if p.speed_kmh > 0 else ""
            badge_text = f"#{p.track_id}  {spd_text}".strip()

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            font_thick = 1
            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)

            # Badge pill geometry
            pad_x = 7
            pad_y = 5
            strip_w = 4  # Team color left strip
            badge_w = tw + pad_x * 2 + strip_w + 2
            badge_h = th + pad_y * 2

            # Position badge above head with margin
            bx1 = center_x - badge_w // 2
            by2 = y1 - 8
            by1 = by2 - badge_h
            bx2 = bx1 + badge_w

            # Keep inside frame boundaries
            if by1 < 4:
                by1 = y2 + 8
                by2 = by1 + badge_h
            bx1 = max(4, min(frame_w - badge_w - 4, bx1))
            bx2 = bx1 + badge_w

            # Semi-transparent dark container overlay
            sub_img = annotated[by1:by2, bx1:bx2]
            dark_bg = np.full(sub_img.shape, (16, 20, 26), dtype=np.uint8)
            cv2.addWeighted(dark_bg, 0.85, sub_img, 0.15, 0, sub_img)
            annotated[by1:by2, bx1:bx2] = sub_img

            # Thin outer border for sharpness
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (65, 75, 90), 1)

            # Team color accent bar on left side of badge
            cv2.rectangle(annotated, (bx1, by1), (bx1 + strip_w, by2), team_col, -1)

            # Crisp white text with drop shadow
            text_x = bx1 + strip_w + pad_x
            text_y = by2 - pad_y
            cv2.putText(annotated, badge_text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)
            cv2.putText(annotated, badge_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        # 2. Annotate Ball
        if tracking_result.ball:
            bx1, by1, bx2, by2 = map(int, tracking_result.ball.bbox)
            bc_x = (bx1 + bx2) // 2
            bc_y = (by1 + by2) // 2
            r = max(5, max(bx2 - bx1, by2 - by1) // 2)
            cv2.circle(annotated, (bc_x, bc_y), r + 2, (0, 255, 255), 2)
            cv2.circle(annotated, (bc_x, bc_y), 3, (0, 0, 255), -1)

        return annotated

    def compose_side_by_side(
        self,
        annotated_frame: np.ndarray,
        radar_frame: np.ndarray,
    ) -> np.ndarray:
        """Compose annotated broadcast frame and 2D radar into a single combined view."""
        h, w = annotated_frame.shape[:2]

        # Resize radar to match height
        radar_h = h
        radar_w = int(self.radar_width * (h / self.radar_height))
        resized_radar = cv2.resize(radar_frame, (radar_w, radar_h))

        combined = np.hstack([annotated_frame, resized_radar])
        return combined
