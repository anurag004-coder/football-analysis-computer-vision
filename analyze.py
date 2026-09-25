"""Universal Football Match Analytics Runner.

Allows running the complete Computer Vision & Tactical Analytics pipeline
on ANY input MP4 football match clip.

Usage:
    python analyze.py path/to/clip.mp4
    python analyze.py path/to/clip.mp4 --team-a "Real Madrid" --team-b "Barcelona"
    python analyze.py path/to/clip.mp4 --team-a "Arsenal" --team-b "Chelsea" --team-a-color "#FF0000" --team-b-color "#0000FF"
    python analyze.py path/to/clip.mp4 --max-frames 500
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import FootballAnalysisPipeline


def infer_teams_from_filename(filename: str) -> tuple:
    """Infer team names from common match video naming conventions (e.g., 'arsenal_vs_chelsea.mp4')."""
    stem = Path(filename).stem.lower()
    # Replace separators with spaces
    cleaned = re.sub(r"[_\-]+", " ", stem)
    if " vs " in cleaned:
        parts = cleaned.split(" vs ", 1)
        team_a = parts[0].strip().title()
        team_b = parts[1].strip().title()
        return team_a, team_b
    elif " v " in cleaned:
        parts = cleaned.split(" v ", 1)
        team_a = parts[0].strip().title()
        team_b = parts[1].strip().title()
        return team_a, team_b
    return "Team A", "Team B"


def update_matches_registry(match_info: dict, registry_file: str = "matches.json"):
    """Register or update analyzed match in matches.json for the web dashboard."""
    registry_path = Path(registry_file)
    matches = []
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                matches = json.load(f)
        except Exception:
            matches = []

    # Update existing match by ID or append
    match_id = match_info.get("id")
    updated = False
    for i, m in enumerate(matches):
        if m.get("id") == match_id:
            matches[i] = match_info
            updated = True
            break
    if not updated:
        matches.append(match_info)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)
    print(f"[Analyze] Match registered in {registry_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Football Match Computer Vision & Tactical Analytics Runner"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input football match video (MP4 file)",
    )
    parser.add_argument(
        "--team-a", "-a",
        default=None,
        type=str,
        help="Name of Team A (auto-inferred from filename if omitted)",
    )
    parser.add_argument(
        "--team-b", "-b",
        default=None,
        type=str,
        help="Name of Team B (auto-inferred from filename if omitted)",
    )
    parser.add_argument(
        "--team-a-color",
        default=None,
        type=str,
        help="Hex color for Team A jersey (e.g. #DC143C for red)",
    )
    parser.add_argument(
        "--team-b-color",
        default=None,
        type=str,
        help="Hex color for Team B jersey (e.g. #FFFFFF for white)",
    )
    parser.add_argument(
        "--max-frames", "-n",
        default=None,
        type=int,
        help="Max frames to process (e.g. 500 frames = 20s @ 25fps)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        type=str,
        help="Output directory (defaults to outputs/<team_a>_vs_<team_b>)",
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        type=str,
        help="YOLOv11 model weights (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        type=str,
        help="Inference device: 'cuda', 'cuda:0', or 'cpu'",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Input video '{args.video}' does not exist.")
        sys.exit(1)

    # Resolve team names
    inferred_a, inferred_b = infer_teams_from_filename(video_path.name)
    team_a = args.team_a or inferred_a
    team_b = args.team_b or inferred_b

    # Resolve output directory
    slug_a = re.sub(r"[^a-zA-Z0-9]+", "_", team_a.lower()).strip("_")
    slug_b = re.sub(r"[^a-zA-Z0-9]+", "_", team_b.lower()).strip("_")
    match_slug = f"{slug_a}_vs_{slug_b}"

    out_dir = Path(args.output) if args.output else Path("outputs") / match_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  TACTICAL FOOTBALL COMPUTER VISION PIPELINE")
    print(f"  Input Clip : {video_path}")
    print(f"  Matchup    : {team_a} vs. {team_b}")
    print(f"  Output Dir : {out_dir}")
    print("=" * 65 + "\n")

    # Initialize and run pipeline
    pipeline = FootballAnalysisPipeline(
        model_path=args.model,
        device=args.device,
        team_a_name=team_a,
        team_b_name=team_b,
        manual_team_a_color=args.team_a_color,
        manual_team_b_color=args.team_b_color,
    )

    report = pipeline.process_video(
        video_path=str(video_path),
        output_dir=str(out_dir),
        max_frames=args.max_frames,
    )

    # Register in matches.json
    stem = video_path.stem
    match_entry = {
        "id": match_slug,
        "name": f"{team_a} vs. {team_b}",
        "team_a": team_a,
        "team_b": team_b,
        "team_a_color": args.team_a_color or "#DC143C",
        "team_b_color": args.team_b_color or "#FFFFFF",
        "frames": report.get("total_frames_processed", 0),
        "fps": report.get("fps", 25.0),
        "duration_sec": round(report.get("total_frames_processed", 0) / max(1.0, report.get("fps", 25.0)), 1),
        "possession": report.get("possession_percentages", {}),
        "total_passes": report.get("total_passes_detected", 0),
        "tactical_metrics": report.get("tactical_metrics", {}),
        "total_tracks": report.get("total_unique_tracks", 0),
        "videos": {
            "combined": f"outputs/{match_slug}/{stem}_combined_web.mp4",
            "annotated": f"outputs/{match_slug}/{stem}_annotated_web.mp4",
            "tactical": f"outputs/{match_slug}/{stem}_tactical_map_web.mp4",
        },
        "pass_images": {
            "team_a": f"outputs/{match_slug}/{stem}_passing_network_{slug_a}.png",
            "team_b": f"outputs/{match_slug}/{stem}_passing_network_{slug_b}.png",
        },
        "report_json": f"outputs/{match_slug}/{stem}_analysis_report.json",
        "physical_csv": f"outputs/{match_slug}/{stem}_player_physical_metrics.csv",
    }
    update_matches_registry(match_entry)

    print("\n" + "=" * 65)
    print("  ANALYSIS COMPLETE!")
    print(f"  Processed Frames : {report.get('total_frames_processed')}")
    print(f"  Possession       : {report.get('possession_percentages')}")
    print(f"  Passes Detected  : {report.get('total_passes_detected')}")
    print(f"  Web Video        : outputs/{match_slug}/{stem}_combined_web.mp4")
    print(f"  Open Dashboard   : http://localhost:8000/dashboard.html")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
