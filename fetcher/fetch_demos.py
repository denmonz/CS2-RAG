"""
fetch_demos.py
--------------
Downloads your N most recent CS2 demos from Leetify into data/demos/.

How it works:
  1. Calls the Leetify public API to fetch your match history
  2. Extracts the demo download URL from each match
  3. Downloads and decompresses each demo (.dem.bz2 or .dem.gz → .dem)
  4. Skips demos that have already been downloaded (unless --redownload)

Prerequisites:
  - Your Steam64 ID  (e.g. 76561198012345678)
  - Optional: a Leetify API key from https://leetify.com/app/developer
    (no key still works but is rate-limited more aggressively)

Usage examples
--------------
# Fetch the 10 most recent demos
python fetcher/fetch_demos.py --steam-id 76561198012345678 --count 10

# Fetch the 5 most recent, using an API key
python fetcher/fetch_demos.py --steam-id 76561198012345678 --count 5 --api-key YOUR_KEY

# Force re-download already-present demos
python fetcher/fetch_demos.py --steam-id 76561198012345678 --count 10 --redownload

# Custom output directory
python fetcher/fetch_demos.py --steam-id 76561198012345678 --out path/to/demos/
"""

import argparse
import bz2
import gzip
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEETIFY_BASE      = "https://api.cs-prod.leetify.com/api"
PROFILE_URL       = LEETIFY_BASE + "/profile/{steam_id}"
MATCHES_URL       = LEETIFY_BASE + "/profile/{steam_id}/mini-matches/cs2"
MATCH_DETAIL_URL  = LEETIFY_BASE + "/games/{game_id}"

DEFAULT_OUT_DIR   = "data/demos"
REQUEST_TIMEOUT   = 30          # seconds per HTTP request
DOWNLOAD_TIMEOUT  = 120         # seconds for large demo files
RETRY_DELAY       = 5           # seconds to wait before retrying after rate-limit


# ---------------------------------------------------------------------------
# Leetify API helpers
# ---------------------------------------------------------------------------

def make_headers(api_key: str | None) -> dict:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def get_match_list(steam_id: str, api_key: str | None) -> list[dict]:
    """Return the full match list for a player from the Leetify API."""
    url     = MATCHES_URL.format(steam_id=steam_id)
    headers = make_headers(api_key)

    print(f"[fetch] Fetching match list for Steam ID {steam_id} ...")
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 429:
        print(f"[fetch] Rate limited — waiting {RETRY_DELAY}s and retrying ...")
        time.sleep(RETRY_DELAY)
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

    resp.raise_for_status()
    data = resp.json()

    # The public API returns either a list directly or a dict with a key
    if isinstance(data, list):
        return data
    for key in ("games", "matches", "data"):
        if key in data and isinstance(data[key], list):
            return data[key]

    raise ValueError(f"Unexpected match list response shape: {list(data.keys())}")


def get_demo_url(game_id: str, api_key: str | None) -> str | None:
    """Fetch a single match's detail page and return the demo download URL."""
    url     = MATCH_DETAIL_URL.format(game_id=game_id)
    headers = make_headers(api_key)

    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return None
    if resp.status_code == 429:
        time.sleep(RETRY_DELAY)
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

    resp.raise_for_status()
    data = resp.json()

    # Leetify stores the demo URL under several possible keys
    for key in ("demoUrl", "demo_url", "downloadUrl", "download_url"):
        val = data.get(key)
        if val:
            return val

    # Also check nested game data
    game = data.get("game") or data.get("gameData") or {}
    for key in ("demoUrl", "demo_url", "downloadUrl", "download_url"):
        val = game.get(key)
        if val:
            return val

    return None


# ---------------------------------------------------------------------------
# Download + decompress
# ---------------------------------------------------------------------------

def decompress_if_needed(compressed_path: Path) -> Path:
    """
    If the file ends in .bz2 or .gz, decompress it in-place and return the
    path of the resulting .dem file.  Returns the input path unchanged
    if no decompression is needed.
    """
    suffix = compressed_path.suffix.lower()

    if suffix == ".bz2":
        out_path = compressed_path.with_suffix("")   # strips .bz2
        print(f"  [decompress] bz2 → {out_path.name}")
        with bz2.open(compressed_path, "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
        compressed_path.unlink()
        return out_path

    if suffix == ".gz":
        out_path = compressed_path.with_suffix("")   # strips .gz
        print(f"  [decompress] gz  → {out_path.name}")
        with gzip.open(compressed_path, "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
        compressed_path.unlink()
        return out_path

    return compressed_path


def download_demo(url: str, out_dir: Path, stem: str) -> Path | None:
    """
    Stream-download a demo file from `url` into `out_dir`.
    Handles .dem, .dem.bz2, and .dem.gz transparently.
    Returns the final .dem path, or None on failure.
    """
    # Derive filename from the URL's last path segment
    url_filename = url.split("?")[0].rstrip("/").split("/")[-1]
    download_path = out_dir / url_filename

    print(f"  [download] {url_filename}")
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(download_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    f.write(chunk)
    except requests.RequestException as exc:
        print(f"  [error] Download failed: {exc}")
        if download_path.exists():
            download_path.unlink()
        return None

    return decompress_if_needed(download_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def already_downloaded(game_id: str, out_dir: Path) -> bool:
    """Return True if a .dem file for this game_id already exists."""
    # We name the final .dem by game_id so the check is deterministic
    return (out_dir / f"{game_id}.dem").exists()


def run(steam_id: str, count: int, out_dir: str,
        api_key: str | None, redownload: bool) -> None:

    demos_dir = Path(out_dir)
    demos_dir.mkdir(parents=True, exist_ok=True)

    # 1 — Fetch match list
    try:
        matches = get_match_list(steam_id, api_key)
    except requests.HTTPError as exc:
        sys.exit(f"[fetch] Failed to fetch match list: {exc}")
    except ValueError as exc:
        sys.exit(f"[fetch] {exc}")

    if not matches:
        sys.exit("[fetch] No matches returned — check your Steam ID.")

    print(f"[fetch] Found {len(matches)} total matches — processing {count} most recent.\n")
    matches = matches[:count]

    downloaded = skipped = failed = 0

    for i, match in enumerate(matches, 1):
        # Leetify uses various key names for the game/match ID
        game_id = (
            match.get("gameId")
            or match.get("game_id")
            or match.get("id")
            or match.get("matchId")
        )
        if not game_id:
            print(f"  [{i}/{count}] Could not find game ID in match entry — skipping.")
            failed += 1
            continue

        map_name   = match.get("mapName") or match.get("map_name") or "unknown"
        played_at  = match.get("gameFinishedAt") or match.get("playedAt") or ""
        label      = f"{played_at[:10]} | {map_name} | {game_id}"

        # Check skip condition
        if not redownload and already_downloaded(game_id, demos_dir):
            print(f"  [{i}/{count}] SKIP    {label}")
            skipped += 1
            continue

        print(f"  [{i}/{count}] Fetching demo URL for: {label}")

        # 2 — Get the demo URL from the match detail endpoint
        try:
            demo_url = get_demo_url(game_id, api_key)
        except requests.HTTPError as exc:
            print(f"  [error] Could not fetch match detail: {exc}")
            failed += 1
            continue

        if not demo_url:
            print(f"  [skip]  No demo URL available for {game_id} (demo may have expired)")
            skipped += 1
            continue

        # 3 — Download + decompress
        result = download_demo(demo_url, demos_dir, stem=game_id)
        if result is None:
            failed += 1
            continue

        # Rename to a stable game_id-based filename if needed
        final_path = demos_dir / f"{game_id}.dem"
        if result != final_path:
            result.rename(final_path)

        print(f"           → saved as {final_path.name}")
        downloaded += 1

        # Brief pause to be polite to the API
        time.sleep(0.5)

    print(
        f"\n[fetch] Done — "
        f"{downloaded} downloaded, {skipped} skipped, {failed} failed "
        f"(total: {count})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Download your N most recent CS2 demos from Leetify.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--steam-id",
        required=True,
        help="Your Steam64 ID (e.g. 76561198012345678). "
             "Find it at steamid.io or in your Leetify profile URL.",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of most recent demos to download (default: 10)",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"Directory to save .dem files (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("LEETIFY_API_KEY"),
        help="Leetify API key for higher rate limits. "
             "Also reads from LEETIFY_API_KEY env var. "
             "Get one at https://leetify.com/app/developer",
    )
    ap.add_argument(
        "--redownload",
        action="store_true",
        help="Re-download demos that are already present in --out",
    )

    args = ap.parse_args()
    run(
        steam_id=args.steam_id,
        count=args.count,
        out_dir=args.out,
        api_key=args.api_key,
        redownload=args.redownload,
    )
