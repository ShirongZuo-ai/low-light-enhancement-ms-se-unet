#!/usr/bin/env python3
"""
generate_hero_video_minimax.py
==============================
Use the MiniMax Pay-as-you-go API (MiniMax-Hailuo-2.3) to generate a
cinematic hero background video for the low-light enhancement landing page.

Output: assets/hero_memory_light.mp4

Set MINIMAX_API_KEY in your environment before running:
    export MINIMAX_API_KEY="your-key-here"
    python3 scripts/generate_hero_video_minimax.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_PATH = ASSETS_DIR / "hero_memory_light.mp4"

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io").rstrip("/")
CREATE_URL = f"{API_BASE}/v1/video_generation"
QUERY_URL = f"{API_BASE}/v1/query/video_generation"
FILE_RETRIEVE_URL = f"{API_BASE}/v1/files/retrieve"

MODEL = "MiniMax-Hailuo-2.3"
DURATION = 6
RESOLUTION = "1080P"
POLL_INTERVAL_S = 10  # seconds between status checks
MAX_WAIT_S = 600  # 10 minutes timeout

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT = (
    "Generate a cinematic full-screen hero background video for a luxury AI photo restoration landing page. "
    "Scene: A warm minimal photo restoration studio with ivory walls, soft beige tones, and several elegant "
    "vintage photographs placed on a clean surface. One dim old photograph is gradually illuminated by a "
    "soft golden light, revealing subtle details. The camera slowly orbits around the scene with smooth "
    "premium motion. "
    "Style: High-end editorial campaign, Apple-like minimalism, ESTUDIO ANONIMO inspired composition, "
    "luxury interior design mood, warm ivory, beige, light gray, charcoal accents, cinematic depth of field, "
    "subtle film grain, elegant lighting, lots of negative space. "
    "Motion: Slow smooth camera orbit, no fast movement, no zoom rush, no sudden cuts, no shake. "
    "The light gently spreads across the old photo. The scene should feel calm, emotional, refined, "
    "expensive, and suitable for a website hero background. "
    "Requirements: 6 seconds, seamless loop feel, no text, no logos, no people, no hands, "
    "no subtitles, no UI elements, no watermark, no distorted objects, enough clean negative space "
    "for overlay text, warm calm premium minimal."
    "[Pan right]"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    """Return a masked version of the API key for safe logging."""
    if len(key) <= 8:
        return "***"
    return key[:4] + "****" + key[-4:]


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Validate environment ---
    api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not api_key:
        print("ERROR: MINIMAX_API_KEY environment variable is not set.", file=sys.stderr)
        print("       Set it with: export MINIMAX_API_KEY=\"your-key-here\"", file=sys.stderr)
        sys.exit(1)

    print(f"MiniMax API base:       {API_BASE}")
    print(f"Model:                  {MODEL}")
    print(f"Duration:               {DURATION}s")
    print(f"Resolution:             {RESOLUTION}")
    print(f"API key:                {_mask_key(api_key)}")
    print(f"Output path:            {OUTPUT_PATH}")
    print()

    # Ensure assets directory exists
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    headers = _auth_headers(api_key)

    # -----------------------------------------------------------------------
    # Step 1: Create the text-to-video generation task
    # -----------------------------------------------------------------------
    print("[1/4] Creating text-to-video task ...")
    create_payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "duration": DURATION,
        "resolution": RESOLUTION,
        "prompt_optimizer": True,
    }

    try:
        resp = requests.post(CREATE_URL, headers=headers, json=create_payload, timeout=30)
    except requests.RequestException as exc:
        print(f"ERROR: Failed to reach {CREATE_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    if resp.status_code != 200:
        print(f"ERROR: Create task returned HTTP {resp.status_code}", file=sys.stderr)
        print(f"Response: {resp.text}", file=sys.stderr)
        sys.exit(1)

    create_data = resp.json()
    base_resp = create_data.get("base_resp", {})
    if base_resp.get("status_code", -1) != 0:
        print(f"ERROR: Create task failed — status_code={base_resp.get('status_code')}", file=sys.stderr)
        print(f"       status_msg={base_resp.get('status_msg')}", file=sys.stderr)
        print(f"Full response: {create_data}", file=sys.stderr)
        sys.exit(1)

    task_id = create_data.get("task_id")
    if not task_id:
        print(f"ERROR: No task_id in response: {create_data}", file=sys.stderr)
        sys.exit(1)

    print(f"       Task created: task_id={task_id}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Poll for completion
    # -----------------------------------------------------------------------
    print(f"[2/4] Polling for completion (every {POLL_INTERVAL_S}s, timeout {MAX_WAIT_S}s) ...")
    started_at = time.monotonic()
    last_status = ""

    while True:
        elapsed = time.monotonic() - started_at
        if elapsed > MAX_WAIT_S:
            print(f"ERROR: Timed out after {elapsed:.0f}s waiting for task {task_id}.", file=sys.stderr)
            sys.exit(1)

        try:
            status_resp = requests.get(
                QUERY_URL,
                headers=headers,
                params={"task_id": task_id},
                timeout=30,
            )
        except requests.RequestException as exc:
            print(f"WARNING: Status check failed ({exc}), retrying ...")
            time.sleep(POLL_INTERVAL_S)
            continue

        if status_resp.status_code != 200:
            print(f"WARNING: Status check returned HTTP {status_resp.status_code}, retrying ...")
            time.sleep(POLL_INTERVAL_S)
            continue

        status_data = status_resp.json()
        status = status_data.get("status", "")
        poll_base = status_data.get("base_resp", {})
        poll_code = poll_base.get("status_code", -1)

        if status != last_status:
            elapsed_str = f"{elapsed:.0f}s"
            print(f"       [{elapsed_str}] status={status} (code={poll_code})")
            last_status = status

        if status == "Success":
            file_id = status_data.get("file_id")
            if not file_id:
                print("ERROR: Task succeeded but no file_id returned.", file=sys.stderr)
                print(f"Full response: {status_data}", file=sys.stderr)
                sys.exit(1)
            print(f"       Task complete! file_id={file_id}")
            print()
            break

        if status == "Fail":
            print("ERROR: Task failed.", file=sys.stderr)
            print(f"Full response: {status_data}", file=sys.stderr)
            sys.exit(1)

        if poll_code in (1002,):
            print(f"       Rate limited (code={poll_code}), backing off ...")
            time.sleep(POLL_INTERVAL_S * 2)
            continue

        if poll_code in (1004,):
            print("ERROR: Authentication failed — check your MINIMAX_API_KEY.", file=sys.stderr)
            sys.exit(1)

        if poll_code in (1026, 1027):
            print(f"ERROR: Content moderation flagged this task (code={poll_code}).", file=sys.stderr)
            print(f"       The prompt or generated video may contain restricted content.", file=sys.stderr)
            sys.exit(1)

        time.sleep(POLL_INTERVAL_S)

    # -----------------------------------------------------------------------
    # Step 3: Retrieve download URL via file API
    # -----------------------------------------------------------------------
    print("[3/4] Retrieving download URL from file API ...")
    try:
        file_resp = requests.get(
            FILE_RETRIEVE_URL,
            headers=headers,
            params={"file_id": file_id},
            timeout=30,
        )
    except requests.RequestException as exc:
        print(f"ERROR: Failed to reach {FILE_RETRIEVE_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    if file_resp.status_code != 200:
        print(f"ERROR: File retrieve returned HTTP {file_resp.status_code}", file=sys.stderr)
        print(f"Response: {file_resp.text}", file=sys.stderr)
        sys.exit(1)

    file_data = file_resp.json()
    file_obj = file_data.get("file", {})
    download_url = file_obj.get("download_url")

    if not download_url:
        print("ERROR: No download_url in file response.", file=sys.stderr)
        print(f"Full response: {file_data}", file=sys.stderr)
        sys.exit(1)

    filename = file_obj.get("filename", "unknown")
    file_bytes = file_obj.get("bytes", 0)
    print(f"       filename:  {filename}")
    print(f"       size:      {file_bytes} bytes ({file_bytes / 1024 / 1024:.1f} MB)")
    print(f"       url:       {download_url[:80]}...")
    print()

    # -----------------------------------------------------------------------
    # Step 4: Download video
    # -----------------------------------------------------------------------
    print(f"[4/4] Downloading video to {OUTPUT_PATH} ...")
    try:
        dl_resp = requests.get(download_url, timeout=120, stream=True)
        dl_resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    total_size = int(dl_resp.headers.get("content-length", 0))
    downloaded = 0
    with open(OUTPUT_PATH, "wb") as f:
        for chunk in dl_resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded * 100 // total_size
                sys.stdout.write(f"\r       {downloaded}/{total_size} bytes ({pct}%)")
                sys.stdout.flush()

    print()
    actual_size = OUTPUT_PATH.stat().st_size
    print(f"       Saved {actual_size} bytes to {OUTPUT_PATH}")
    print()
    print("Done! Hero video saved.")
    print(f"Restart app_v3.py to use it:  python3 app_v3.py")


if __name__ == "__main__":
    main()
