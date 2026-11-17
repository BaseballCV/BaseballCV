"""Export the S6 v11 findings table: one row per validated transit.

Joins run1/s6_transits.csv to the game_sync manifest + result pitches (the
same lag-windowed match as spike_s6_join_transits.py: nearest unused PA
event with transit - db_video_s in [40, 180] s, ties toward +100 s), and
adds batter names (mojibake-repaired), pitch info, wall clock, and derived
metrics. Output is UTF-8 with BOM so Excel renders accented names.

Usage: python scripts/export_s6_findings.py [OUT_CSV]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

CLOCK0 = 66764.0
DB_OFFSET = 5400.0
LAG_LO, LAG_HI, ANCHOR = 40.0, 180.0, 100.0
FTS_TO_MPH = 0.681818
# events that never send a batter-runner down the 1B line; excluded as match
# candidates (they share db times with adjacent PAs and would steal their rows)
NON_TRANSIT_EVENTS = {"Pitching Substitution", "Offensive Substitution",
                      "Defensive Switch", "Wild Pitch", "Stolen Base 2B",
                      "Offensive Switch"}


def fix_name(s: str) -> str:
    """Repair utf-8 text that was decoded as latin-1 upstream."""
    if not isinstance(s, str):
        return s
    try:
        fixed = s.encode("latin-1").decode("utf-8")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def wall_clock(video_s: float) -> str:
    w = CLOCK0 + video_s
    return f"{int(w // 3600) % 24:02d}:{int(w // 60) % 60:02d}:{int(w) % 60:02d}"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\Carlos Marcano\Downloads\s6_findings_game869992.csv"
    tr = pl.read_csv("run1/s6_transits.csv").sort("t_start_s")
    m = pl.read_csv("run1/game_sync/manifest.csv").sort("video_start_s")
    p = pl.read_parquet("run1/game_sync/pbp_cache/game_869992.parquet.pitches")
    p = p.with_columns(
        (pl.col("time").str.strptime(pl.Time, "%H:%M:%S").cast(pl.Int64) / 1e9
         + DB_OFFSET - CLOCK0).alias("db")).sort("no")

    events = []
    for r in m.iter_rows(named=True):
        cand = p.filter((pl.col("db") >= r["video_start_s"] - 30)
                        & (pl.col("db") <= r["video_end_s"] + 30))
        res = cand.filter(pl.col("play_result") != "")
        row = res if len(res) else cand
        if len(row):
            events.append({
                "event_seq": r["event_seq"],
                "inning": r["inning"],
                "half": "T" if r["is_top_inning"] else "B",
                "event": r["event_text"],
                "batter": fix_name(r["batter_name"]),
                "pitch_no": int(row["no"].to_list()[-1]),
                "pitch_result": str(row["play_result"].to_list()[-1]),
                "db": float(row["db"].to_list()[-1]),
                "db_raw": str(row["time"].to_list()[-1]),
            })

    used: set[int] = set()
    rows = []
    for k, t in enumerate(tr.iter_rows(named=True), start=1):
        best, best_d = None, 1e9
        for i, e in enumerate(events):
            if i in used or e["event"] in NON_TRANSIT_EVENTS:
                continue
            lag = t["t_start_s"] - e["db"]
            if LAG_LO <= lag <= LAG_HI and abs(lag - ANCHOR) < best_d:
                best, best_d = i, abs(lag - ANCHOR)
        dur = t["t_end_s"] - t["t_start_s"]
        dist = t["x_end_ft"] - t["x_start_ft"]
        r = {
            "transit_no": k,
            "wall_clock": wall_clock(t["t_start_s"]),
            "video_t_start_s": round(t["t_start_s"], 2),
            "video_t_end_s": round(t["t_end_s"], 2),
            "duration_s": round(dur, 2),
            "x_start_ft": round(t["x_start_ft"], 1),
            "x_end_ft": round(t["x_end_ft"], 1),
            "distance_ft": round(dist, 1),
            "speed_ft_s": round(t["speed_ft_s"], 2),
            "speed_mph": round(t["speed_ft_s"] * FTS_TO_MPH, 2),
            "n_points": t["n_pts"],
            "est_home_to_1b_s_const_speed":
                round(dur * 90.0 / dist, 2) if dist > 0 else None,
            "matched_event": "yes" if best is not None else "no",
            "event_seq": None, "inning": None, "half": None, "event": None,
            "batter": None, "pitch_no": None, "pitch_result": None,
            "pitch_db_raw_time": None, "db_lag_s": None,
        }
        if best is not None:
            used.add(best)
            e = events[best]
            r.update({
                "event_seq": e["event_seq"], "inning": e["inning"],
                "half": e["half"], "event": e["event"], "batter": e["batter"],
                "pitch_no": e["pitch_no"], "pitch_result": e["pitch_result"],
                "pitch_db_raw_time": e["db_raw"],
                "db_lag_s": round(t["t_start_s"] - e["db"], 1),
            })
        rows.append(r)

    cols = list(rows[0].keys())
    df = pl.DataFrame(rows, schema=cols, orient="row")
    data = df.write_csv()
    with open(out, "wb") as f:
        f.write("\ufeff".encode("utf-8") + data.encode("utf-8"))

    matched = df.filter(pl.col("matched_event") == "yes")
    sp = matched["speed_ft_s"]
    print(f"{len(df)} transits -> {out}")
    print(f"matched {len(matched)} to PA events; "
          f"speed median {np.median(sp.to_numpy()):.1f} ft/s "
          f"({np.median(sp.to_numpy()) * FTS_TO_MPH:.1f} mph), "
          f"range {sp.min():.1f}-{sp.max():.1f} ft/s")
    by = matched.group_by("event").agg(
        pl.len().alias("n"), pl.col("speed_ft_s").mean().round(1).alias("mean_ft_s"))
    print(by.sort("n", descending=True))


if __name__ == "__main__":
    main()
