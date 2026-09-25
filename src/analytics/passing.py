"""Passing Network and Possession Analytics for Football Computer Vision.

Detects possession states, pass events, turnovers, and renders tactical
passing networks and team formation shapes on standard FIFA 105m x 68m pitches.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


@dataclass
class PassEvent:
    """Represents a completed pass between two teammates."""
    pass_id: int
    team: str
    from_track_id: int
    to_track_id: int
    start_frame: int
    end_frame: int
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    distance_meters: float


class PassingNetworkAnalytics:
    """Detects passing events, computes possession, and visualizes passing networks."""

    def __init__(
        self,
        possession_radius_meters: float = 4.0,
        min_flight_frames: int = 2,
        team_a_name: str = "Team A",
        team_b_name: str = "Team B",
        team_a_color: str = "#DC143C",
        team_b_color: str = "#FFFFFF",
    ):
        """Initialize passing analytics.

        Args:
            possession_radius_meters: Maximum distance between player and ball to establish possession.
            min_flight_frames: Minimum frames ball must travel between players to count as a pass.
            team_a_name: Name of first team.
            team_b_name: Name of second team.
            team_a_color: Primary hex color of Team A.
            team_b_color: Primary hex color of Team B.
        """
        self.possession_radius = possession_radius_meters
        self.min_flight_frames = min_flight_frames
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
        self.team_a_color = team_a_color
        self.team_b_color = team_b_color

        # State tracking
        self.current_possessor: Optional[int] = None
        self.current_possessor_team: Optional[str] = None
        self.last_possession_frame: int = 0
        self.last_possession_pos: Optional[Tuple[float, float]] = None

        self.pass_events: List[PassEvent] = []
        self.possession_counts: Dict[str, int] = {self.team_a_name: 0, self.team_b_name: 0, "Contested": 0}
        self.player_avg_positions: Dict[int, List[Tuple[float, float]]] = {}
        self.player_teams: Dict[int, str] = {}
        self.player_frame_counts: Dict[int, int] = {}

    def update_frame(
        self,
        frame_idx: int,
        ball_pos: Optional[Tuple[float, float]],
        player_positions: Dict[int, Tuple[float, float]],  # track_id -> (x, y) meters
        player_teams: Dict[int, str],                      # track_id -> team_str
    ):
        """Update possession state and detect passes for current frame."""
        # 1. Update player metadata & coordinates
        for tid, pos in player_positions.items():
            if tid not in self.player_avg_positions:
                self.player_avg_positions[tid] = []
            self.player_avg_positions[tid].append(pos)
            self.player_frame_counts[tid] = self.player_frame_counts.get(tid, 0) + 1
            if tid in player_teams:
                self.player_teams[tid] = player_teams[tid]

        # 2. Check Ball Possession
        if ball_pos is not None and player_positions:
            bx, by = ball_pos
            closest_player_id = None
            min_dist = float("inf")

            for tid, (px, py) in player_positions.items():
                dist = np.hypot(px - bx, py - by)
                if dist < min_dist:
                    min_dist = dist
                    closest_player_id = tid

            if closest_player_id is not None and min_dist <= self.possession_radius:
                team = player_teams.get(closest_player_id, "Unknown")
                if team in self.possession_counts:
                    self.possession_counts[team] += 1

                if self.current_possessor is None:
                    # Establish initial possession
                    self.current_possessor = closest_player_id
                    self.current_possessor_team = team
                    self.last_possession_frame = frame_idx
                    self.last_possession_pos = player_positions[closest_player_id]

                elif self.current_possessor != closest_player_id:
                    # Possession transition
                    flight_time = frame_idx - self.last_possession_frame
                    if flight_time >= self.min_flight_frames:
                        # Same team transition = Completed Pass
                        if team == self.current_possessor_team and team in (self.team_a_name, self.team_b_name):
                            start_pos = self.last_possession_pos or player_positions[closest_player_id]
                            end_pos = player_positions[closest_player_id]
                            dist_m = float(np.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]))

                            # Ignore negligible shifts (< 2.0 meters)
                            if dist_m >= 2.0:
                                self.pass_events.append(PassEvent(
                                    pass_id=len(self.pass_events) + 1,
                                    team=team,
                                    from_track_id=self.current_possessor,
                                    to_track_id=closest_player_id,
                                    start_frame=self.last_possession_frame,
                                    end_frame=frame_idx,
                                    start_pos=start_pos,
                                    end_pos=end_pos,
                                    distance_meters=round(dist_m, 2),
                                ))

                    # Update current possessor
                    self.current_possessor = closest_player_id
                    self.current_possessor_team = team
                    self.last_possession_frame = frame_idx
                    self.last_possession_pos = player_positions[closest_player_id]
            else:
                # Ball is in flight / unattached
                if self.current_possessor_team in self.possession_counts:
                    self.possession_counts[self.current_possessor_team] += 1
                else:
                    self.possession_counts["Contested"] += 1
        else:
            # Ball position not available, maintain active team control if known
            if self.current_possessor_team in self.possession_counts:
                self.possession_counts[self.current_possessor_team] += 1
            else:
                self.possession_counts["Contested"] += 1

    def get_possession_percentages(self) -> Dict[str, float]:
        """Compute team ball possession percentages."""
        count_a = self.possession_counts.get(self.team_a_name, 0)
        count_b = self.possession_counts.get(self.team_b_name, 0)
        total = count_a + count_b
        if total == 0:
            return {self.team_a_name: 50.0, self.team_b_name: 50.0}
        pct_a = round((count_a / total) * 100.0, 1)
        pct_b = round(100.0 - pct_a, 1)
        return {self.team_a_name: pct_a, self.team_b_name: pct_b}

    def get_team_tactical_metrics(self, team: str) -> Dict[str, Union[float, int]]:
        """Calculate tactical width, length, and passing metrics for a team."""
        team_players = [
            tid for tid, t in self.player_teams.items()
            if t == team and self.player_frame_counts.get(tid, 0) >= 15
        ]
        xs, ys = [], []
        for tid in team_players:
            pts = self.player_avg_positions.get(tid, [])
            if pts:
                xs.append(float(np.median([p[0] for p in pts])))
                ys.append(float(np.median([p[1] for p in pts])))

        width_m = round(float(np.ptp(ys)), 1) if len(ys) > 1 else 0.0
        depth_m = round(float(np.ptp(xs)), 1) if len(xs) > 1 else 0.0
        passes = [p for p in self.pass_events if p.team == team]

        return {
            "team_width_m": width_m,
            "team_depth_m": depth_m,
            "total_passes": len(passes),
            "active_players": len(team_players),
        }

    def build_passing_matrix(self, team: str) -> pd.DataFrame:
        """Create passing matrix showing pass counts between all player pairs for a team."""
        team_passes = [p for p in self.pass_events if p.team == team]
        team_players = sorted([
            tid for tid, t in self.player_teams.items()
            if t == team and self.player_frame_counts.get(tid, 0) >= 15
        ])

        matrix = pd.DataFrame(0, index=team_players, columns=team_players)
        for p in team_passes:
            if p.from_track_id in matrix.index and p.to_track_id in matrix.columns:
                matrix.loc[p.from_track_id, p.to_track_id] += 1
        return matrix

    def plot_passing_network(
        self,
        team: str,
        output_path: str,
        team_color: Optional[str] = None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        """Render high-resolution tactical passing network diagram on FIFA pitch layout."""
        if team_color is None:
            team_color = self.team_a_color if team == self.team_a_name else self.team_b_color

        # Filter active players for this team (at least 15 frames tracked)
        team_players = [
            tid for tid, t in self.player_teams.items()
            if t == team and self.player_frame_counts.get(tid, 0) >= 15
        ]

        # Compute median positional centroids
        node_positions: Dict[int, Tuple[float, float]] = {}
        for tid in team_players:
            points = self.player_avg_positions.get(tid, [])
            if points:
                mx = float(np.median([pt[0] for pt in points]))
                my = float(np.median([pt[1] for pt in points]))
                # Clamp within pitch boundaries
                mx = max(3.0, min(pitch_length - 3.0, mx))
                my = max(3.0, min(pitch_width - 3.0, my))
                node_positions[tid] = (mx, my)

        # Sort and pick top 11 most persistent players to form the starting tactical structure
        sorted_players = sorted(
            node_positions.keys(),
            key=lambda tid: self.player_frame_counts.get(tid, 0),
            reverse=True
        )[:11]
        active_nodes = {tid: node_positions[tid] for tid in sorted_players}

        team_passes = [
            p for p in self.pass_events
            if p.team == team and p.from_track_id in active_nodes and p.to_track_id in active_nodes
        ]

        # Calculate luminance of team_color to pick optimal text and border contrast
        color_hex = team_color.lstrip("#")
        try:
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
        except Exception:
            lum = 200

        text_color = "#101418" if lum > 145 else "#FFFFFF"
        border_color = "#101418" if lum > 145 else "#FFFFFF"

        # Initialize Plot
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor("#0b1017")
        ax.set_facecolor("#164a28")  # Premium deep pitch turf green

        # Alternating turf grass stripes
        stripe_width = pitch_length / 10.0
        for s in range(10):
            if s % 2 == 0:
                ax.add_patch(patches.Rectangle(
                    (s * stripe_width, 0), stripe_width, pitch_width,
                    facecolor="#1a562f", edgecolor="none", zorder=1
                ))

        # Pitch Boundary & Markings
        line_color = "#FFFFFF"
        ax.plot([0, pitch_length, pitch_length, 0, 0], [0, 0, pitch_width, pitch_width, 0], color=line_color, lw=2.2, zorder=2)
        ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color=line_color, lw=1.8, zorder=2)

        # Center Circle & Spot
        center_circle = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, color=line_color, fill=False, lw=1.8, zorder=2)
        ax.add_patch(center_circle)
        ax.plot(pitch_length / 2, pitch_width / 2, "o", color=line_color, markersize=4, zorder=2)

        # Penalty Areas
        # Left Box (16.5m x 40.3m)
        box_y1 = (pitch_width - 40.3) / 2
        box_y2 = (pitch_width + 40.3) / 2
        ax.plot([0, 16.5, 16.5, 0], [box_y1, box_y1, box_y2, box_y2], color=line_color, lw=1.8, zorder=2)
        # Left Goal Area (5.5m x 18.3m)
        g_y1 = (pitch_width - 18.3) / 2
        g_y2 = (pitch_width + 18.3) / 2
        ax.plot([0, 5.5, 5.5, 0], [g_y1, g_y1, g_y2, g_y2], color=line_color, lw=1.5, zorder=2)

        # Right Box
        r_box_x = pitch_length - 16.5
        ax.plot([pitch_length, r_box_x, r_box_x, pitch_length], [box_y1, box_y1, box_y2, box_y2], color=line_color, lw=1.8, zorder=2)
        # Right Goal Area
        r_g_x = pitch_length - 5.5
        ax.plot([pitch_length, r_g_x, r_g_x, pitch_length], [g_y1, g_y1, g_y2, g_y2], color=line_color, lw=1.5, zorder=2)

        # Draw Pass Edges (Directed Links)
        edge_counts: Dict[Tuple[int, int], int] = {}
        for p in team_passes:
            pair = (p.from_track_id, p.to_track_id)
            edge_counts[pair] = edge_counts.get(pair, 0) + 1

        for (u, v), count in edge_counts.items():
            if u in active_nodes and v in active_nodes:
                x1, y1 = active_nodes[u]
                x2, y2 = active_nodes[v]
                lw = min(count * 2.5 + 1.0, 7.0)
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#00E5FF",
                        lw=lw,
                        mutation_scale=16,
                        alpha=0.85,
                        shrinkA=14,
                        shrinkB=14,
                    ),
                    zorder=4,
                )

        # Draw All Active Player Nodes (Tactical Centroids)
        for tid, (x, y) in active_nodes.items():
            # Size proportional to tracked activity
            activity_score = self.player_frame_counts.get(tid, 20)
            node_size = min(900, max(500, int(activity_score * 1.5)))

            # Glow outer ring
            ax.scatter(x, y, s=node_size + 150, color="#000000", alpha=0.5, zorder=5)
            # Main Team Node
            ax.scatter(
                x, y, s=node_size, color=team_color,
                edgecolors=border_color, lw=2.5, zorder=6
            )
            # High-contrast Player ID Label
            ax.text(
                x, y, str(tid), color=text_color,
                fontsize=11, ha="center", va="center", weight="bold", zorder=7
            )

        # Invert Y to align top-left origin with broadcast camera convention
        ax.set_xlim(0, pitch_length)
        ax.set_ylim(pitch_width, 0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title & Tactical Stat Header
        total_p = len(team_passes)
        pos_pct = self.get_possession_percentages().get(team, 50.0)
        title_str = f"{team} · Tactical Passing Network & Positional Shape"
        sub_str = f"Possession: {pos_pct}%   |   Passes Detected: {total_p}   |   Active Players Plotted: {len(active_nodes)}"

        fig.text(0.5, 0.95, title_str, color="#F0F4F8", fontsize=15, weight="bold", ha="center")
        fig.text(0.5, 0.915, sub_str, color="#00E5FF", fontsize=11, fontfamily="monospace", ha="center")

        plt.subplots_adjust(top=0.88, bottom=0.04, left=0.03, right=0.97)
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
