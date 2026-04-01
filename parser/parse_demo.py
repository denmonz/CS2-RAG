"""
parse_demo.py
-------------
Parses a CS2 .dem file using awpy and outputs structured round-level
chunks as JSON, ready for embedding into ChromaDB.
"""

import json
import argparse
from pathlib import Path
from awpy import Demo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def build_round_chunk(round_num: int, round_data: dict, meta: dict) -> dict:
    """
    Convert a single round's raw data into a rich text chunk + metadata dict.

    The text field is what gets embedded — make it as descriptive as possible
    so semantic search can find relevant rounds later.
    """
    outcome    = round_data.get("winner_side", "unknown")
    reason     = round_data.get("reason", "unknown")
    ct_eq      = round_data.get("ct_eq_val", 0)
    t_eq       = round_data.get("t_eq_val", 0)
    ct_spend   = round_data.get("ct_spend", 0)
    t_spend    = round_data.get("t_spend", 0)
    kills      = round_data.get("kills", [])
    bomb       = round_data.get("bomb_planted", False)

    # ---- Summarise kills ---------------------------------------------------
    kill_lines = []
    for k in kills:
        attacker = k.get("attacker_name", "unknown")
        victim   = k.get("victim_name", "unknown")
        weapon   = k.get("weapon", "unknown")
        headshot = "headshot" if k.get("headshot") else "bodyshot"
        kill_lines.append(f"{attacker} killed {victim} with {weapon} ({headshot})")

    kill_summary = "; ".join(kill_lines) if kill_lines else "no kills recorded"

    # ---- Prose description (this is what gets embedded) --------------------
    text = (
        f"Round {round_num} on {meta['map']} — "
        f"Winner: {outcome} via {reason}. "
        f"CT equipment value ${ct_eq:,} (spent ${ct_spend:,}), "
        f"T equipment value ${t_eq:,} (spent ${t_spend:,}). "
        f"Bomb planted: {bomb}. "
        f"Kills this round: {kill_summary}."
    )

    return {
        "chunk_id": f"{meta['match_id']}_round_{round_num:03d}",
        "text":     text,
        "metadata": {
            "match_id":    meta["match_id"],
            "map":         meta["map"],
            "round":       round_num,
            "winner_side": outcome,
            "end_reason":  reason,
            "bomb_planted": bomb,
            "ct_eq_val":   ct_eq,
            "t_eq_val":    t_eq,
            "ct_spend":    ct_spend,
            "t_spend":     t_spend,
            "kill_count":  len(kills),
        },
    }


# ---------------------------------------------------------------------------
# Main parsing logic
# ---------------------------------------------------------------------------

def parse_demo(demo_path: str, output_dir: str = "data/parsed") -> list[dict]:
    """
    Parse a .dem file and return a list of round-level chunk dicts.
    Also writes the result to <output_dir>/<match_id>.json.
    """
    demo_path = Path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo not found: {demo_path}")

    print(f"[parser] Loading demo: {demo_path.name} ...")
    demo = Demo(path=str(demo_path))

    # awpy exposes DataFrames; pull what we need
    rounds_df = demo.rounds       # one row per round
    kills_df  = demo.kills        # one row per kill event
    map_name  = demo.header.get("map_name", "unknown")
    match_id  = demo_path.stem    # use filename as match identifier

    meta = {"match_id": match_id, "map": map_name}

    chunks: list[dict] = []

    for _, row in rounds_df.iterrows():
        rnum = int(row.get("round_num", 0))

        # Gather kills that belong to this round
        round_kills_df = kills_df[kills_df["round_num"] == rnum] if not kills_df.empty else []
        kills_list = []
        for _, k in (round_kills_df.iterrows() if hasattr(round_kills_df, "iterrows") else []):
            kills_list.append({
                "attacker_name": k.get("attacker_name", "unknown"),
                "victim_name":   k.get("victim_name",   "unknown"),
                "weapon":        k.get("weapon",         "unknown"),
                "headshot":      bool(k.get("headshot", False)),
            })

        round_data = {
            "winner_side": row.get("winner_side", "unknown"),
            "reason":      row.get("reason",      "unknown"),
            "ct_eq_val":   int(row.get("ct_eq_val",  0) or 0),
            "t_eq_val":    int(row.get("t_eq_val",   0) or 0),
            "ct_spend":    int(row.get("ct_spend",   0) or 0),
            "t_spend":     int(row.get("t_spend",    0) or 0),
            "bomb_planted": bool(row.get("bomb_planted", False)),
            "kills":       kills_list,
        }

        chunks.append(build_round_chunk(rnum, round_data, meta))

    # Write to disk
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{match_id}.json"
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"[parser] Parsed {len(chunks)} rounds → {out_path}")
    return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse a CS2 demo into round chunks.")
    ap.add_argument("demo",       help="Path to .dem file")
    ap.add_argument("--out", default="data/parsed", help="Output directory")
    args = ap.parse_args()

    chunks = parse_demo(args.demo, args.out)
    print(f"[parser] Done. Sample chunk:\n{json.dumps(chunks[0], indent=2)}")
