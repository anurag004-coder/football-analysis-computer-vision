"""Tactical Football Analytics Dashboard Server.

Serves the interactive web dashboard with video streaming support (HTTP Range requests)
and exposes REST API endpoints to dynamically query analyzed matches and trigger
new clip analyses directly from the browser.

Usage:
    python serve_dashboard.py
"""

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
import urllib.parse
import webbrowser

import sys

PORT = int(os.environ.get("PORT", 8000))
if len(sys.argv) > 1:
    for i, arg in enumerate(sys.argv):
        if arg in ("--port", "-p") and i + 1 < len(sys.argv):
            PORT = int(sys.argv[i + 1])

WORKSPACE_DIR = Path(__file__).parent.resolve()
ACTIVE_ANALYSIS = {"running": False, "progress": "Idle", "log": ""}


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE_DIR), **kwargs)

    def end_headers(self):
        # Enable Range requests for smooth video scrubbing in browsers
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/matches":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            registry_file = WORKSPACE_DIR / "matches.json"
            if registry_file.exists():
                with open(registry_file, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(b"[]")
            return
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(ACTIVE_ANALYSIS).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            video_path = data.get("video_path")
            if not video_path:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "video_path is required"}')
                return

            if ACTIVE_ANALYSIS["running"]:
                self.send_response(409)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "An analysis is already running in background"}')
                return

            # Start analysis in background thread
            thread = threading.Thread(target=run_analysis_worker, args=(data,))
            thread.daemon = True
            thread.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "started", "message": "Analysis started in background"}')
            return

        super().do_POST()


def run_analysis_worker(data: dict):
    global ACTIVE_ANALYSIS
    ACTIVE_ANALYSIS["running"] = True
    ACTIVE_ANALYSIS["progress"] = "Starting AI pipeline..."
    ACTIVE_ANALYSIS["log"] = ""

    video_path = data.get("video_path")
    team_a = data.get("team_a")
    team_b = data.get("team_b")
    team_a_color = data.get("team_a_color")
    team_b_color = data.get("team_b_color")
    max_frames = data.get("max_frames")

    python_exe = sys.executable
    cmd = [python_exe, "analyze.py", video_path]
    if team_a:
        cmd.extend(["--team-a", team_a])
    if team_b:
        cmd.extend(["--team-b", team_b])
    if team_a_color:
        cmd.extend(["--team-a-color", team_a_color])
    if team_b_color:
        cmd.extend(["--team-b-color", team_b_color])
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(WORKSPACE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in proc.stdout:
            ACTIVE_ANALYSIS["log"] += line
            if "Pass 1" in line:
                ACTIVE_ANALYSIS["progress"] = "Pass 1: Tracking players & ball..."
            elif "Pass 2" in line:
                ACTIVE_ANALYSIS["progress"] = "Pass 2: Pitch homography & tactical rendering..."
            elif "Clustering" in line:
                ACTIVE_ANALYSIS["progress"] = "Clustering jersey colors..."
            elif "Web transcode" in line or "Transcoding" in line:
                ACTIVE_ANALYSIS["progress"] = "Transcoding H.264 video for web..."
        proc.wait()
        ACTIVE_ANALYSIS["progress"] = "Complete!"
    except Exception as e:
        ACTIVE_ANALYSIS["progress"] = f"Error: {e}"
    finally:
        ACTIVE_ANALYSIS["running"] = False


def serve():
    global PORT
    while PORT < 8020:
        try:
            with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
                url = f"http://localhost:{PORT}/dashboard.html"
                print("\n" + "=" * 60)
                print("  TACTICAL FOOTBALL ANALYTICS DASHBOARD")
                print(f"  Serving at: {url}")
                print("  API Endpoints: /api/matches, /api/analyze, /api/status")
                print("  Press Ctrl+C to stop server")
                print("=" * 60 + "\n")
                httpd.serve_forever()
        except OSError:
            PORT += 1


if __name__ == "__main__":
    serve()
