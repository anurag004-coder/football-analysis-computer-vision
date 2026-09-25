"""Download open research sample football match clip.

Downloads the standard open-access football tracking research clip
into data/samples/sample_match.mp4.
"""

import os
from pathlib import Path
import urllib.request
import requests


def download_file_from_google_drive(file_id: str, destination: Path):
    """Download large or small file from Google Drive handling confirm token."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"[Downloader] Sample video already exists: {destination} ({destination.stat().st_size / (1024*1024):.1f} MB)")
        return str(destination)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print(f"[Downloader] Fetching sample research clip from Google Drive (ID: {file_id})...")
    response = session.get(url, params={"id": file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    chunk_size = 32768
    total_downloaded = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                total_downloaded += len(chunk)

    print(f"[Downloader] Download completed: {destination} ({total_downloaded / (1024*1024):.1f} MB)")
    return str(destination)


if __name__ == "__main__":
    target = Path("data/samples/sample_match.mp4")
    # Standard open research tracking clip ID (from DFL Bundesliga tracking tutorial)
    drive_id = "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
    try:
        download_file_from_google_drive(drive_id, target)
    except Exception as e:
        print(f"[Downloader] Note: {e}")
