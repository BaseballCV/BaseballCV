"""Full game_metrics run on the verified 869992 sync artifacts (runner phase).

Loads run1/game_sync artifacts, tracks every clip, writes game_metrics.csv.
"""
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseballcv.functions import BaseballTools

res = BaseballTools(device="cpu").compute_game_metrics(
    sync_results_dir="run1/game_sync", results_dir="run1/game_metrics")

print("=== SUMMARY ===")
print(json.dumps(res["summary"], indent=1))
df = res["metrics"]
print(df.group_by("status").len().sort("len", descending=True))
ok = df.filter(pl.col("status") == "ok")
if len(ok):
    print(ok.select(["event_seq", "event_type", "time_to_1b_s", "peak_speed_ft_s",
                     "speed_at_30_ft", "route_efficiency"]).head(15))
