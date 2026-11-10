"""Rule-based runner categories and per-runner aggregation."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .config import StealConfig


def label_lead_length(lead_ft: Optional[float], cfg: StealConfig) -> Optional[str]:
    if lead_ft is None or pd.isna(lead_ft):
        return None
    if lead_ft < cfg.lead_short_ft:
        return "short"
    return "long" if lead_ft >= cfg.lead_long_ft else "medium"


def label_lead_style(pre_pitch_speed: Optional[float], cfg: StealConfig) -> Optional[str]:
    if pre_pitch_speed is None or pd.isna(pre_pitch_speed):
        return None
    return "walking" if pre_pitch_speed >= cfg.walking_speed_ft_s else "static"


def label_jump(reaction_s: Optional[float], cfg: StealConfig) -> Optional[str]:
    if reaction_s is None or pd.isna(reaction_s):
        return None
    if reaction_s < cfg.jump_early_s:
        return "early"
    return "late" if reaction_s > cfg.jump_late_s else "on_time"


def categorize_play(row: Dict, cfg: Optional[StealConfig] = None) -> Dict[str, Optional[str]]:
    """Category labels for one play's metric row.

    Distance-based labels (lead length) are withheld when ``scale_confidence`` is below
    ``cfg.min_scale_conf``; timing labels do not depend on scale and are still produced.
    """
    cfg = cfg or StealConfig()
    conf = row.get("scale_confidence")
    distance_ok = conf is not None and not pd.isna(conf) and conf >= cfg.min_scale_conf
    labels = {
        "lead_length": label_lead_length(row.get("primary_lead_ft"), cfg) if distance_ok else None,
        "lead_style": label_lead_style(row.get("pre_pitch_speed_ft_s"), cfg) if distance_ok else None,
        "jump": label_jump(row.get("reaction_time_s"), cfg),
        "first_step": row.get("first_step") or None,
    }
    parts = [labels[k] for k in ("lead_length", "lead_style", "jump", "first_step") if labels[k]]
    labels["profile"] = "-".join(parts) if parts else None
    return labels


def _mode(s: pd.Series) -> Optional[str]:
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def aggregate_runners(df: pd.DataFrame, group_col: str = "runner_id", min_plays: int = 1) -> pd.DataFrame:
    """Per-runner means, spreads, and most common labels from a per-play metrics table."""
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    numeric = ["primary_lead_ft", "secondary_lead_ft", "lead_gain_ft", "reaction_time_s",
               "time_to_10ft_s", "peak_speed_ft_s", "accel_0_5s_ft_s2"]
    numeric = [c for c in numeric if c in df.columns]
    label_cols = [c for c in ("lead_length", "lead_style", "jump", "first_step") if c in df.columns]
    rows = []
    for key, g in df.groupby(group_col):
        if len(g) < min_plays:
            continue
        row = {group_col: key, "n_plays": len(g)}
        for c in numeric:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std() if g[c].notna().sum() > 1 else None
        for c in label_cols:
            row[f"{c}_mode"] = _mode(g[c])
            n = g[c].notna().sum()
            row[f"{c}_consistency"] = (g[c].value_counts().iloc[0] / n) if n else None
        parts = [row[f"{c}_mode"] for c in label_cols if row.get(f"{c}_mode")]
        row["profile"] = "-".join(parts) if parts else None
        rows.append(row)
    return pd.DataFrame(rows)
