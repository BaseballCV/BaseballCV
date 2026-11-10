"""Find stolen-base attempts of second (with Savant play IDs) from the MLB statsapi live feed."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import polars as pl

from baseballcv.utilities import BaseballCVLogger
from .crawler import requests_with_retry, sanitize_date_range
from .gameday import GAMEDAY_PBP_URL, _get_team, _parse_game_dates

logger = BaseballCVLogger.get_logger("steal_events")

# statsapi event codes for attempts at second base
STEAL_2B_EVENTS = {"stolen_base_2b": "SB", "caught_stealing_2b": "CS", "pickoff_caught_stealing_2b": "CS"}
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people/{}"


def parse_height_in(height: Optional[str]) -> Optional[float]:
    """Convert statsapi heights like ``6' 1"`` to inches."""
    if not height:
        return None
    m = re.match(r"\s*(\d+)'\s*(\d+)", height)
    return float(int(m.group(1)) * 12 + int(m.group(2))) if m else None


def parse_steal_events(play_by_play: Dict, game_pk: int) -> List[Dict]:
    """Extract 1B->2B steal attempts from one game's playByPlay JSON.

    Only events that carry an ``actionPlayId`` (i.e. have a Savant video) are returned.
    """
    out: List[Dict] = []
    for play in play_by_play.get("allPlays", []):
        matchup = play.get("matchup", {})
        names = {r.get("details", {}).get("runner", {}).get("id"): r.get("details", {}).get("runner", {}).get("fullName")
                 for r in play.get("runners", [])}
        for ev in play.get("playEvents", []):
            code = (ev.get("details", {}).get("eventType") or "")
            if code not in STEAL_2B_EVENTS or not ev.get("actionPlayId"):
                continue
            runner_id = (ev.get("player") or {}).get("id")
            out.append({
                "game_pk": game_pk,
                "play_id": ev["actionPlayId"],
                "result": STEAL_2B_EVENTS[code],
                "event": code,
                "inning": play.get("about", {}).get("inning"),
                "half_inning": play.get("about", {}).get("halfInning"),
                "outs": ev.get("count", {}).get("outs"),
                "balls": ev.get("count", {}).get("balls"),
                "strikes": ev.get("count", {}).get("strikes"),
                "runner_id": runner_id,
                "runner_name": names.get(runner_id),
                "pitcher_id": matchup.get("pitcher", {}).get("id"),
                "pitcher_name": matchup.get("pitcher", {}).get("fullName"),
                "p_throws": matchup.get("pitchHand", {}).get("code"),
            })
    return out


def get_runner_height_in(runner_id: int) -> Optional[float]:
    """Roster height in inches for a player, or None if unavailable."""
    resp = requests_with_retry(PEOPLE_URL.format(runner_id))
    if resp is None:
        return None
    people = resp.json().get("people", [])
    return parse_height_in(people[0].get("height")) if people else None


def get_steal_events(start_dt: str, end_dt: Optional[str] = None, team_abbr: Optional[str] = None,
                     player: Optional[int] = None, max_events: Optional[int] = None) -> pl.DataFrame:
    """Steal attempts of second base in a date range, ready for ``BaseballSavVideoScraper.from_play_ids``.

    Args:
        start_dt: Start date (YYYY-MM-DD).
        end_dt: End date. Defaults to ``start_dt``.
        team_abbr: Only games involving this team.
        player: Only attempts by this runner ID.
        max_events: Cap on returned rows (random sample).
    """
    start, end = sanitize_date_range(start_dt, end_dt)
    team = _get_team(team_abbr, None, end.year)
    games = _parse_game_dates(start, end, team)
    rows: List[Dict] = []
    for game_pk in games:
        resp = requests_with_retry(GAMEDAY_PBP_URL.format(game_pk))
        if resp is None:
            logger.warning(f"No play-by-play for game {game_pk}")
            continue
        rows.extend(parse_steal_events(resp.json(), game_pk))
    if player is not None:
        rows = [r for r in rows if r["runner_id"] == player]
    df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"game_pk": pl.Int64, "play_id": pl.Utf8})
    if max_events and len(df) > max_events:
        df = df.sample(max_events)
    return df
