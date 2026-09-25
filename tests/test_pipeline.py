"""Unit and Integration Tests for Football Analytics Computer Vision Pipeline.

Verifies:
- Standard FIFA pitch model and keypoint coordinates
- Homography projection and reprojection error
- Jersey color extraction and team clustering
- Movement metrics (speed, distance calculation)
- Bounded Voronoi space control partition
- Possession tracking and passing network heuristics
"""

import numpy as np
import pytest

from src.pitch.pitch_model import PitchModel, PitchDimensions
from src.pitch.homography import PitchHomography
from src.pitch.calibrator import PitchCalibrator
from src.classify.classifier import TeamClassifier
from src.analytics.metrics import MovementAnalytics
from src.analytics.voronoi import VoronoiSpaceControl
from src.analytics.passing import PassingNetworkAnalytics


def test_pitch_model_geometry():
    """Verify standard FIFA pitch dimensions and keypoint landmark positions."""
    pitch = PitchModel()
    assert pitch.dim.length == 105.0
    assert pitch.dim.width == 68.0

    # Center spot should be at (52.5, 34.0)
    center = pitch.get_keypoint("center_spot")
    assert center == (52.5, 34.0)

    # Corners
    assert pitch.get_keypoint("top_left_corner") == (0.0, 0.0)
    assert pitch.get_keypoint("bottom_right_corner") == (105.0, 68.0)

    # Inside pitch check
    assert pitch.is_inside_pitch(52.5, 34.0) is True
    assert pitch.is_inside_pitch(-5.0, 34.0) is False


def test_homography_exact_projection():
    """Test homography computation and verify reprojection error is near 0 for clean points."""
    img_pts = np.array([
        [100.0, 100.0],
        [900.0, 100.0],
        [950.0, 600.0],
        [50.0, 600.0],
    ])
    pitch_pts = np.array([
        [0.0, 0.0],
        [105.0, 0.0],
        [105.0, 68.0],
        [0.0, 68.0],
    ])

    homography = PitchHomography()
    H, error = homography.compute_from_points(img_pts, pitch_pts)

    assert H is not None
    # Reprojection error on exact correspondence should be < 0.1 px (PRD target is < 3.0 px)
    assert error < 0.1

    # Project forward and back
    transformed = homography.image_to_pitch(img_pts)
    np.testing.assert_allclose(transformed, pitch_pts, atol=1e-2)

    reprojected = homography.pitch_to_image(pitch_pts)
    np.testing.assert_allclose(reprojected, img_pts, atol=1e-2)


def test_ground_point_from_bbox():
    """Verify player feet ground contact point extraction from bounding box."""
    bbox = [100.0, 50.0, 200.0, 250.0]  # [x1, y1, x2, y2]
    feet_x, feet_y = PitchHomography.bbox_to_ground_point(bbox)
    assert feet_x == 150.0  # Center x
    assert feet_y == 250.0  # Bottom y


def test_movement_analytics():
    """Verify distance accumulation and speed calculation."""
    fps = 30.0
    analytics = MovementAnalytics(fps=fps, smoothing_window=3)

    # Player moving 10 meters along X in 1 second (30 frames) -> 10 m/s = 36 km/h
    track_id = 1
    for frame in range(31):
        x = frame * (10.0 / 30.0)
        y = 34.0
        analytics.add_position(track_id, frame, x, y)

    metrics_df = analytics.compute_all_metrics()
    assert len(metrics_df) == 1
    row = metrics_df.iloc[0]

    assert row["track_id"] == 1
    np.testing.assert_allclose(row["total_distance_m"], 10.0, atol=0.1)
    # 10 m/s = 36.0 km/h
    np.testing.assert_allclose(row["avg_speed_kmh"], 36.0, atol=1.5)


def test_voronoi_space_control():
    """Verify Voronoi pitch partition and space dominance percentages."""
    voronoi_engine = VoronoiSpaceControl()

    # Place 3 Team A players on left side, 3 Team B players on right side
    points = np.array([
        [20.0, 20.0],
        [20.0, 48.0],
        [35.0, 34.0],
        [70.0, 20.0],
        [70.0, 48.0],
        [85.0, 34.0],
    ])
    teams = ["Team A", "Team A", "Team A", "Team B", "Team B", "Team B"]

    res = voronoi_engine.compute_frame_voronoi(points, teams)
    assert "team_control_pct" in res
    pct_a = res["team_control_pct"]["Team A"]
    pct_b = res["team_control_pct"]["Team B"]

    # Space percentages should sum to 100%
    assert abs((pct_a + pct_b) - 100.0) < 0.5
    # Symmetrical placement should yield roughly 50-50 space control
    assert 40.0 <= pct_a <= 60.0


def test_passing_network_detection():
    """Verify passing heuristic: possession transfer between teammates registers a pass."""
    passing = PassingNetworkAnalytics(possession_radius_meters=2.0, min_flight_frames=3)

    player_teams = {10: "Team A", 11: "Team A", 20: "Team B"}
    player_positions = {
        10: (30.0, 30.0),
        11: (50.0, 30.0),
        20: (40.0, 50.0),
    }

    # Frame 0 to 5: Ball with player 10
    for f in range(6):
        passing.update_frame(f, ball_pos=(30.5, 30.5), player_positions=player_positions, player_teams=player_teams)

    # Frame 6 to 9: Ball in flight towards player 11
    passing.update_frame(6, ball_pos=(35.0, 30.0), player_positions=player_positions, player_teams=player_teams)
    passing.update_frame(7, ball_pos=(40.0, 30.0), player_positions=player_positions, player_teams=player_teams)
    passing.update_frame(8, ball_pos=(45.0, 30.0), player_positions=player_positions, player_teams=player_teams)

    # Frame 10: Ball arrives at player 11
    passing.update_frame(10, ball_pos=(50.2, 30.1), player_positions=player_positions, player_teams=player_teams)

    assert len(passing.pass_events) == 1
    completed_pass = passing.pass_events[0]
    assert completed_pass.from_track_id == 10
    assert completed_pass.to_track_id == 11
    assert completed_pass.team == "Team A"
    np.testing.assert_allclose(completed_pass.distance_meters, 20.0, atol=0.5)


def test_team_classifier_kmeans():
    """Verify jersey color clustering separates two distinct color sets."""
    classifier = TeamClassifier()

    # Red jersey frame crop (BGR)
    red_crop = np.zeros((60, 40, 3), dtype=np.uint8)
    red_crop[:, :] = [0, 0, 220]  # Red in BGR

    # Blue jersey frame crop (BGR)
    blue_crop = np.zeros((60, 40, 3), dtype=np.uint8)
    blue_crop[:, :] = [220, 0, 0]  # Blue in BGR

    # Accumulate samples for track 1 (Red) and track 2 (Blue)
    for _ in range(5):
        classifier.track_color_samples.setdefault(1, []).append(
            TeamClassifier.extract_dominant_color_hsv(red_crop)
        )
        classifier.track_color_samples.setdefault(2, []).append(
            TeamClassifier.extract_dominant_color_hsv(blue_crop)
        )

    classifier.fit_team_clusters()

    team_1 = classifier.get_track_team(1)
    team_2 = classifier.get_track_team(2)

    # Tracks with distinct colors must be assigned to different teams
    assert team_1 != team_2
