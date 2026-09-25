"""Synthetic Football Broadcast Video Generator.

Generates stand-in broadcast football match footage (MP4)
for development, automated testing, and CI/CD without copyright concerns.
Complies with PRD Section 4 (Data Sourcing & Rights).
"""

import math
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


class SyntheticMatchGenerator:
    """Generates synthetic broadcast football video with realistic player movements and passing."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

    def generate(self, output_path: str, duration_sec: int = 5):
        """Generate synthetic MP4 video."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        total_frames = int(duration_sec * self.fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Pitch layout parameters
        pitch_top = int(0.20 * self.height)
        pitch_bot = int(0.90 * self.height)
        pitch_left = int(0.08 * self.width)
        pitch_right = int(0.92 * self.width)
        pitch_center_x = (pitch_left + pitch_right) // 2
        pitch_center_y = (pitch_top + pitch_bot) // 2

        # Initialize Players: (x, y, vx, vy, team, radius, jersey_color_bgr)
        players = []

        # Team A (Red kits: BGR [30, 30, 210])
        team_a_color = (30, 30, 210)
        for i in range(8):
            px = pitch_left + int((0.15 + 0.30 * (i % 3)) * (pitch_right - pitch_left))
            py = pitch_top + int((0.20 + 0.25 * (i // 3)) * (pitch_bot - pitch_top))
            players.append({
                "x": float(px), "y": float(py),
                "vx": np.random.uniform(-1.5, 1.5), "vy": np.random.uniform(-1.0, 1.0),
                "team": "Team A", "color": team_a_color, "id": i + 1,
            })

        # Team B (Blue kits: BGR [210, 100, 20])
        team_b_color = (210, 100, 20)
        for i in range(8):
            px = pitch_center_x + int((0.05 + 0.35 * (i % 3)) * (pitch_right - pitch_center_x))
            py = pitch_top + int((0.20 + 0.25 * (i // 3)) * (pitch_bot - pitch_top))
            players.append({
                "x": float(px), "y": float(py),
                "vx": np.random.uniform(-1.5, 1.5), "vy": np.random.uniform(-1.0, 1.0),
                "team": "Team B", "color": team_b_color, "id": i + 10,
            })

        # Ball State
        ball = {
            "x": float(pitch_center_x - 50),
            "y": float(pitch_center_y),
            "vx": 3.0, "vy": 1.2,
        }

        for frame_idx in range(total_frames):
            frame = np.full((self.height, self.width, 3), (35, 110, 45), dtype=np.uint8)

            # Pitch boundary lines (White)
            cv2.rectangle(frame, (pitch_left, pitch_top), (pitch_right, pitch_bot), (240, 240, 240), 2)
            cv2.line(frame, (pitch_center_x, pitch_top), (pitch_center_x, pitch_bot), (240, 240, 240), 2)
            cv2.circle(frame, (pitch_center_x, pitch_center_y), 70, (240, 240, 240), 2)
            cv2.circle(frame, (pitch_center_x, pitch_center_y), 4, (240, 240, 240), -1)

            # Penalty boxes
            pen_w = int(0.16 * (pitch_right - pitch_left))
            pen_h = int(0.50 * (pitch_bot - pitch_top))
            pen_top = pitch_center_y - pen_h // 2
            pen_bot = pitch_center_y + pen_h // 2
            cv2.rectangle(frame, (pitch_left, pen_top), (pitch_left + pen_w, pen_bot), (240, 240, 240), 2)
            cv2.rectangle(frame, (pitch_right - pen_w, pen_top), (pitch_right, pen_bot), (240, 240, 240), 2)

            # Update & Draw Players
            for p in players:
                p["x"] += p["vx"]
                p["y"] += p["vy"]

                # Bounce off pitch boundaries
                if p["x"] <= pitch_left + 20 or p["x"] >= pitch_right - 20:
                    p["vx"] *= -1
                if p["y"] <= pitch_top + 20 or p["y"] >= pitch_bot - 20:
                    p["vy"] *= -1

                ix, iy = int(p["x"]), int(p["y"])

                # Draw realistic player representation (head, jersey torso, shorts, legs)
                # Player height ~ 36px, width ~ 14px
                # Head
                cv2.circle(frame, (ix, iy - 26), 5, (190, 180, 170), -1)
                # Jersey Torso
                cv2.rectangle(frame, (ix - 8, iy - 21), (ix + 8, iy - 7), p["color"], -1)
                # Shorts
                cv2.rectangle(frame, (ix - 7, iy - 7), (ix + 7, iy), (230, 230, 230), -1)
                # Legs & Boots
                cv2.line(frame, (ix - 4, iy), (ix - 4, iy + 10), (190, 180, 170), 2)
                cv2.line(frame, (ix + 4, iy), (ix + 4, iy + 10), (190, 180, 170), 2)

            # Update & Draw Ball
            ball["x"] += ball["vx"]
            ball["y"] += ball["vy"]
            if ball["x"] <= pitch_left + 10 or ball["x"] >= pitch_right - 10:
                ball["vx"] *= -1
            if ball["y"] <= pitch_top + 10 or ball["y"] >= pitch_bot - 10:
                ball["vy"] *= -1

            bx, by = int(ball["x"]), int(ball["y"])
            # Ball (White circle with dark seam)
            cv2.circle(frame, (bx, by), 5, (255, 255, 255), -1)
            cv2.circle(frame, (bx, by), 5, (40, 40, 40), 1)

            writer.write(frame)

        writer.release()
        print(f"[SyntheticGenerator] Video generated: {output_path} ({total_frames} frames)")
        return output_path


if __name__ == "__main__":
    gen = SyntheticMatchGenerator()
    gen.generate("data/samples/synthetic_match.mp4", duration_sec=5)
