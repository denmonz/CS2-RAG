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
import polars as pl


# ---------------------------------------------------------------------------
# Chunk builder
# ---------------------------------------------------------------------------

def build_round_chunk(round_num: int, round_data: dict, meta: dict) -> dict:
    """
    Convert a single round's raw data into a rich text chunk + metadata dict.

    The text field is what gets embedded — make it as descriptive as possible
    so semantic search can find relevant rounds later.
    """
    outcome  = round_data.get("winner_side", "unknown")
    reason   = round_data.get("reason",      "unknown")
    ct_eq    = round_data.get("ct_eq_val",   0)
    t_eq     = round_data.get("t_eq_val",    0)
    ct_spend = round_data.get("ct_spend",    0)
    t_spend  = round_data.get("t_spend",     0)
    kills    = round_data.get("kills",       [])
    bomb     = round_data.get("bomb_planted", False)

    kill_lines = []
    for k in kills:
        attacker = k.get("attacker_name", "unknown")
        victim   = k.get("victim_name",   "unknown")
        weapon   = k.get("weapon",        "unknown")
        headshot = "headshot" if k.get("headshot") else "bodyshot"
        kill_lines.append(f"{attacker} killed {victim} with {weapon} ({headshot})")

    kill_summary = "; ".join(kill_lines) if kill_lines else "no kills recorded"

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
            "match_id":     meta["match_id"],
            "map":          meta["map"],
            "round":        round_num,
            "winner_side":  outcome,
            "end_reason":   reason,
            "bomb_planted": bomb,
            "ct_eq_val":    ct_eq,
            "t_eq_val":     t_eq,
            "ct_spend":     ct_spend,
            "t_spend":      t_spend,
            "kill_count":   len(kills),
        },
    }


# ---------------------------------------------------------------------------
# Single-demo parser
# ---------------------------------------------------------------------------

def parse_demo(demo_path: Path, output_dir: Path, reparse: bool) -> list[dict] | None:
    """
    Parse one .dem file and write <output_dir>/<stem>.json.

    Returns the list of chunks on success, or None if the file was skipped
    because it was already parsed and reparse=False.

    awpy returns polars.DataFrames, so we use:
      - .iter_rows(named=True)  instead of .iterrows()
      - .filter(pl.col(…) == v) instead of df[df[col] == v]
      - .is_empty()             instead of .empty
    """
    out_path = output_dir / f"{demo_path.stem}.json"

    if out_path.exists() and not reparse:
        print(f"  [skip]   {demo_path.name} → already parsed ({out_path.name})")
        return None

    print(f"  [parse]  {demo_path.name} ...")
    demo = Demo(path=str(demo_path))
    demo.parse()

    rounds_df: pl.DataFrame = demo.rounds
    kills_df:  pl.DataFrame = demo.kills
    map_name = demo.header.get("map_name", "unknown")
    match_id = demo_path.stem

    meta   = {"match_id": match_id, "map": map_name}
    chunks: list[dict] = []

    for row in rounds_df.iter_rows(named=True):
        rnum = int(row.get("round_num") or 0)

        kills_list: list[dict] = []
        if not kills_df.is_empty() and "round_num" in kills_df.columns:
            for k in kills_df.filter(pl.col("round_num") == rnum).iter_rows(named=True):
                kills_list.append({
                    "attacker_name": k.get("attacker_name") or "unknown",
                    "victim_name":   k.get("victim_name")   or "unknown",
                    "weapon":        k.get("weapon")        or "unknown",
                    "headshot":      bool(k.get("headshot") or False),
                })

        round_data = {
            "winner_side":  row.get("winner_side")  or "unknown",
            "reason":       row.get("reason")       or "unknown",
            "ct_eq_val":    int(row.get("ct_eq_val")  or 0),
            "t_eq_val":     int(row.get("t_eq_val")   or 0),
            "ct_spend":     int(row.get("ct_spend")   or 0),
            "t_spend":      int(row.get("t_spend")    or 0),
            "bomb_planted": bool(row.get("bomb_planted") or False),
            "kills":        kills_list,
        }

        chunks.append(build_round_chunk(rnum, round_data, meta))

    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"           → {len(chunks)} rounds written to {out_path}")
    return chunks


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def resolve_demos(target: str | None) -> list[Path]:
    """
    Return a sorted list of .dem files to process.

    - target is None          → scan data/demos/
    - target is a directory   → scan that directory
    - target is a .dem file   → return just that file
    """
    if target is None:
        base = Path("data/demos")
    else:
        base = Path(target)

    if base.is_dir():
        demos = sorted(base.glob("*.dem"))
        if not demos:
            raise FileNotFoundError(f"No .dem files found in directory: {base}")
        return demos

    if base.is_file() and base.suffix == ".dem":
        return [base]

    raise ValueError(
        f"'{target}' is not a .dem file or a directory containing .dem files."
    )


def run(target: str | None, output_dir: str, reparse: bool) -> None:
    demos = resolve_demos(target)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total   = len(demos)
    parsed  = 0
    skipped = 0
    failed  = 0

    print(f"\n[parser] Found {total} demo(s) to process  "
          f"(reparse={'yes' if reparse else 'no'})\n")

    for demo_path in demos:
        try:
            result = parse_demo(demo_path, out_dir, reparse)
            if result is None:
                skipped += 1
            else:
                parsed += 1
        except Exception as exc:
            print(f"  [error]  {demo_path.name}: {exc}")
            failed += 1

    print(
        f"\n[parser] Done — "
        f"{parsed} parsed, {skipped} skipped, {failed} failed  "
        f"(total: {total})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Parse CS2 .dem files into round-level JSON chunks.\n\n"
            "TARGET can be:\n"
            "  • a single .dem file\n"
            "  • a directory of .dem files\n"
            "  • omitted (defaults to data/demos/)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Path to a .dem file or directory of demos (default: data/demos/)",
    )
    ap.add_argument(
        "--out",
        default="data/parsed",
        help="Directory to write parsed JSON files (default: data/parsed/)",
    )
    ap.add_argument(
        "--reparse",
        action="store_true",
        help="Re-parse and overwrite demos that have already been parsed",
    )
    args = ap.parse_args()
    run(target=args.target, output_dir=args.out, reparse=args.reparse)
