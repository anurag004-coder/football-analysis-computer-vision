"""Football CV Pipeline CLI.

Usage per PRD specifications:
    python pipeline.py --input clip.mp4 --output outputs/
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import FootballAnalysisPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Football Analytics Computer Vision Pipeline (YOLOv11 + ByteTrack + Homography + Tactical Analytics)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="Path to input football match video (MP4)",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        type=str,
        help="Directory to save output videos, CSVs, Parquet, and reports (default: outputs/)",
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        type=str,
        help="Path or name of YOLOv11 model weights (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        type=str,
        help="Inference device: 'cuda', 'cuda:0', 'cpu' (default: auto-detected)",
    )
    parser.add_argument(
        "--max-frames",
        default=None,
        type=int,
        help="Maximum number of frames to process (useful for rapid testing / benchmarking)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch interactive GUI window to select pitch landmarks on the first frame",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        type=str,
        help="Path to precomputed calibration JSON file",
    )
    parser.add_argument(
        "--team-a-name",
        default="Team A",
        type=str,
        help="Display name for Team A (e.g. 'VfB Stuttgart' or 'Borussia Mgladbach')",
    )
    parser.add_argument(
        "--team-b-name",
        default="Team B",
        type=str,
        help="Display name for Team B (e.g. 'Greuther Furth' or 'VfL Wolfsburg')",
    )
    parser.add_argument(
        "--team-a-color",
        default=None,
        type=str,
        help="Optional hex color seed for Team A (e.g. #FF0000)",
    )
    parser.add_argument(
        "--team-b-color",
        default=None,
        type=str,
        help="Optional hex color seed for Team B (e.g. #0000FF)",
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Disable rendering of annotated broadcast video",
    )
    parser.add_argument(
        "--no-tactical-map",
        action="store_true",
        help="Disable rendering of 2D tactical radar video",
    )
    parser.add_argument(
        "--no-side-by-side",
        action="store_true",
        help="Disable rendering of combined side-by-side video",
    )

    args = parser.parse_args()

    pipeline = FootballAnalysisPipeline(
        model_path=args.model,
        device=args.device,
        calibration_file=args.calibration_file,
        team_a_name=args.team_a_name,
        team_b_name=args.team_b_name,
        manual_team_a_color=args.team_a_color,
        manual_team_b_color=args.team_b_color,
    )

    pipeline.process_video(
        video_path=args.input,
        output_dir=args.output,
        max_frames=args.max_frames,
        render_annotated=not args.no_annotated,
        render_tactical_map=not args.no_tactical_map,
        render_side_by_side=not args.no_side_by_side,
        interactive_calibration=args.calibrate,
    )


if __name__ == "__main__":
    main()
