"""S6 v11: join transit chains to PBP events, measure the db-clock offset.

Reads run1/s6_transits.csv (from spike_s6_transit_scan.py) and the game_sync
manifest + result pitches, then matches each PA event to its nearest UNUSED
transit in a lag window around the empirical db offset (+100 s in video_s
terms; burned-clock fit says DB_OFFSET ~5505, not the 5400 guess). A plain
order-zip is wrong here: 73 transits vs 103 manifest rows (pitching subs,
walks, strikeouts and caught flyouts produce no full 24->80 ft transit) make
order alignment slip ~+3000 s by inning 8.

Round finding this script pinned: the lag is CONSTANT at ~+100 s across all
9 innings — the db clock does not drift; every db-derived video_s simply
needs ~+100 s. See docs/steal_analysis_handoff.md S6 v11.

Usage: python scripts/spike_s6_join_transits.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

CLOCK0 = 66764.0
DB_OFFSET = 5400.0     # game_sync's guess; empirically ~105 s short
LAG_LO, LAG_HI = 40.0, 180.0   # accepted transit - db_video_s window
ANCHOR = 100.0         # expected lag; ties break toward this


def main():
    tr = pl.read_csv("run1/s6_transits.csv")
    m = pl.read_csv("run1/game_sync/manifest.csv").sort("video_start_s")
    p = pl.read_parquet("run1/game_sync/pbp_cache/game_869992.parquet.pitches")
    p = p.with_columns(
        (pl.col("time").str.strptime(pl.Time, "%H:%M:%S").cast(pl.Int64) / 1e9
         + DB_OFFSET - CLOCK0).alias("db_video_s")
    ).sort("no")

    # one db time per manifest event: last pitch with a play_result in window
    evs = []
    for r in m.iter_rows(named=True):
        cand = p.filter(
            (pl.col("db_video_s") >= r["video_start_s"] - 30)
            & (pl.col("db_video_s") <= r["video_end_s"] + 30)
        )
        res = cand.filter(pl.col("play_result") != "")
        row = res if len(res) else cand
        if len(row) == 0:
            continue
        evs.append({
            "inning": r["inning"],
            "half": "T" if r["is_top_inning"] else "B",
            "event": str(r["event_text"]),
            "db": float(row["db_video_s"].to_list()[-1]),
        })

    ts = tr["t_start_s"].to_numpy()
    spd = tr["speed_ft_s"].to_numpy()
    used: set[int] = set()
    lags = []
    print(f"{len(evs)} events, {len(tr)} transits\n")
    print(f"{'event':>22} {'db':>7} {'transit':>8} {'lag':>6} {'spd':>5}")
    for e in evs:
        best = None
        for i, t in enumerate(ts):
            if i in used:
                continue
            lag = t - e["db"]
            if LAG_LO <= lag <= LAG_HI and (
                best is None or abs(lag - ANCHOR) < abs(best[1] - ANCHOR)
            ):
                best = (i, lag)
        if best is None:
            print(f"{e['event'][:16]:>16} {e['inning']}{e['half']:<1} "
                  f"{e['db']:>7.0f} {'--':>8} {'--':>6}")
            continue
        used.add(best[0])
        lags.append(best[1])
        print(f"{e['event'][:16]:>16} {e['inning']}{e['half']:<1} "
              f"{e['db']:>7.0f} {ts[best[0]]:>8.1f} {best[1]:>+6.0f} "
              f"{spd[best[0]]:>5.1f}")
    a = np.array(lags)
    print(f"\nmatched {len(lags)}, lag median {np.median(a):+.1f}s "
          f"IQR {np.percentile(a, 25):+.0f}..{np.percentile(a, 75):+.0f} "
          f"(equivalent DB_OFFSET ~{DB_OFFSET + np.median(a):.0f})")


if __name__ == "__main__":
    main()
