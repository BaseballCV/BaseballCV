"""Read-only client for the LVBP league SQLite play-by-play DB.

SAFETY CONTRACT (learned from the platform's WAL setup):
- The DB at /var/www/navegantes/baseball_data.db is written by the sync chain
  as `www-data`. A read by any other user leaves that user's -wal/-shm files
  behind and breaks every subsequent sync write. We therefore ALWAYS read via
  `sudo -n -u www-data sqlite3 "file:...?mode=ro"` — mode=ro alone does NOT
  prevent the -wal/-shm creation, running as the owner does.
- Nothing in this module ever opens a write-capable connection.
- Exhibition games (game_type 'E') are present in the tables; filtering is a
  hard requirement, enforced with an error, not a silent drop.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl

from baseballcv.utilities import BaseballCVLogger

from .config import GameSyncConfig

GAME_HEADER_SQL = """
SELECT game_pk, official_date, game_datetime, day_night, game_type, season_stage,
       venue_name, away_team_abbrev, home_team_abbrev, away_runs, home_runs, innings
FROM play_log_games WHERE game_pk = {game_pk}
"""

EVENTS_SQL = """
SELECT event_seq, at_bat_index, inning, is_top_inning, event_type, event_text,
       is_pa, is_ab, batter_id, batter_name, bat_side, pitcher_id, pitcher_name,
       pitch_hand, outs_before, outs_after, r1_before, r2_before, r3_before,
       r1_after, r2_after, r3_after, base_state_before, base_state_after,
       away_score_before, home_score_before, away_score_after, home_score_after,
       balls, strikes
FROM play_log_events WHERE game_pk = {game_pk} ORDER BY event_seq
"""

PITCHES_SQL = """
SELECT no, time, plate_appearance, inning, half_inning, is_top_inning,
       pitcher_name, batter_name, balls, strikes, outs, pitch_type, pitch_call,
       play_result
FROM pitch_by_pitch WHERE game_id = {game_pk} ORDER BY no
"""


@dataclass
class GameSnapshot:
    """One game's PBP rows, materialized locally as polars DataFrames."""
    game_pk: int
    game_date: date
    header: pl.DataFrame
    events: pl.DataFrame
    pitches: pl.DataFrame

    @property
    def game_type(self) -> str:
        return self.header["game_type"][0]

    def play_times(self) -> pl.DataFrame:
        """Per-event DB wall-clock datetimes.

        play_log_events has no time column; the time of an event is taken from
        its plate appearance's first pitch. NOTE (measured on game 869992):
        pitch_by_pitch.plate_appearance is a LINEUP SLOT (1..~10) that keeps
        counting across half-innings (top-2nd ends at pa 7, bottom-2nd starts
        at pa 7) and matches neither play_log_events.at_bat_index (0..79,
        game-wide) nor any arithmetic of it. The stable correspondence is the
        ORDINAL of the PA inside its half-inning, which is chronological on
        both sides. pitch_by_pitch.time is HH:MM:SS on the game's local day —
        anchored to the header date, with a day rollover guard for extra-
        inning games past midnight.
        """
        pitches = self.pitches.filter(pl.col("time").is_not_null())
        pa = (
            pitches.sort("no")
            .group_by(["inning", "is_top_inning", "plate_appearance"])
            .agg(pl.col("time").first().alias("pa_first_pitch_time"),
                 pl.col("time").last().alias("pa_last_pitch_time"))
            .sort(["inning", "is_top_inning", "plate_appearance"])
            .with_columns(pl.col("plate_appearance").rank("dense")
                          .over(["inning", "is_top_inning"]).alias("pa_ordinal"))
            .drop("plate_appearance")
        )
        ev = self.events.with_columns(pl.col("at_bat_index").cast(pl.Int64, strict=False))
        ev_pa = (
            ev.select(["inning", "is_top_inning", "at_bat_index"]).unique()
            .sort(["inning", "is_top_inning", "at_bat_index"])
            .with_columns(pl.col("at_bat_index").rank("dense")
                          .over(["inning", "is_top_inning"]).alias("pa_ordinal"))
        )
        joined = ev.join(ev_pa.join(pa, on=["inning", "is_top_inning", "pa_ordinal"], how="left")
                         .drop("pa_ordinal"),
                         on=["inning", "is_top_inning", "at_bat_index"], how="left")
        base_dt = datetime.combine(self.game_date, datetime.min.time())
        for col in ("pa_first_pitch_time", "pa_last_pitch_time"):
            # HH:MM:SS text -> seconds-of-day -> datetime on the game date
            secs = pl.col(col).str.split(":").list.eval(pl.element().cast(pl.Int64))
            h, m, s = (secs.list.get(0), secs.list.get(1), secs.list.get(2))
            joined = joined.with_columns(
                (base_dt + pl.duration(hours=h, minutes=m, seconds=s)).alias(f"{col}_dt")
            ).drop(col)
        # midnight rollover: an event whose time jumped backwards vs the previous
        # event's belongs to the next day
        times = joined["pa_first_pitch_time_dt"].to_list()
        fixed, day_shift, prev = [], 0, None
        for t in times:
            if t is not None and prev is not None and t < prev - timedelta(minutes=10):
                day_shift += 1
            prev = t if t is not None else prev
            fixed.append(t + timedelta(days=day_shift) if t is not None else None)
        return joined.with_columns(pl.Series("event_time_dt", fixed))


class PBPClient:
    """Runs read-only queries over ssh and caches snapshots as parquet."""

    def __init__(self, config: Optional[GameSyncConfig] = None,
                 cache_dir: str = ".pbp_cache", logger=None) -> None:
        self.config = config or GameSyncConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or BaseballCVLogger.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _query(self, sql: str) -> list[dict]:
        uri = f"file:{self.config.db_path}?mode=ro"
        # ssh joins its argv with spaces and hands them to the remote shell:
        # newlines in the SQL would be executed as shell commands, and any
        # whitespace/quotes must be escaped for the REMOTE shell, hence shlex.
        sql_flat = " ".join(sql.split())
        remote = (f"sudo -n -u {shlex.quote(self.config.db_read_user)} "
                  f"sqlite3 -json {shlex.quote(uri)} {shlex.quote(sql_flat)}")
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            f"{self.config.ssh_user}@{self.config.ssh_host}", remote,
        ]
        assert "mode=ro" in remote  # hard guard: this module never writes
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"DB query failed: {proc.stderr.strip()[:500]}")
        out = proc.stdout.strip()
        return json.loads(out) if out else []

    # ------------------------------------------------------------------
    def fetch_game(self, game_pk: int, refresh: bool = False) -> GameSnapshot:
        cache = self.cache_dir / f"game_{game_pk}.parquet"
        if cache.exists() and not refresh:
            self.logger.info(f"Loading cached snapshot {cache}")
            header = pl.read_parquet(cache)
            return self._snapshot_from_parts(game_pk, header,
                                             pl.read_parquet(str(cache) + ".events"),
                                             pl.read_parquet(str(cache) + ".pitches"))
        header = pl.DataFrame(self._query(GAME_HEADER_SQL.format(game_pk=game_pk)))
        events = pl.DataFrame(self._query(EVENTS_SQL.format(game_pk=game_pk)))
        pitches = pl.DataFrame(self._query(PITCHES_SQL.format(game_pk=game_pk)))
        if header.is_empty():
            raise ValueError(f"game_pk {game_pk} not found in play_log_games")
        if header["game_type"][0] == "E":
            raise ValueError(f"game_pk {game_pk} is an exhibition game (game_type E)")
        header.write_parquet(cache)
        events.write_parquet(str(cache) + ".events")
        pitches.write_parquet(str(cache) + ".pitches")
        self.logger.info(f"Snapshot for game {game_pk}: {len(events)} events, {len(pitches)} pitches")
        return self._snapshot_from_parts(game_pk, header, events, pitches)

    def _snapshot_from_parts(self, game_pk, header, events, pitches) -> GameSnapshot:
        official = header["official_date"][0]  # "YYYY-MM-DD"
        game_date = date.fromisoformat(str(official)[:10])
        return GameSnapshot(game_pk=game_pk, game_date=game_date,
                            header=header, events=events, pitches=pitches)

    def list_games(self, day: str) -> pl.DataFrame:
        """Games on one date (YYYY-MM-DD), regular season only."""
        sql = ("SELECT game_pk, official_date, game_type, away_team_abbrev, home_team_abbrev "
               "FROM play_log_games WHERE official_date LIKE '" + day + "%' AND game_type != 'E'")
        return pl.DataFrame(self._query(sql))
