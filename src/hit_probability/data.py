"""Download Statcast data and turn raw pitches into modeling-ready batted balls."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hit_probability.config import (
    EVENTS_TO_KEEP,
    HIT_EVENTS,
    LEAK_FREE_FEATURES,
    VIDEO_FEATURES,
)

logger = logging.getLogger(__name__)

BUNT_PATTERN = r"\bbunt"

# Approximate home-plate location in Statcast's hc_x / hc_y hit-coordinate system.
HOME_PLATE_X = 125.42
HOME_PLATE_Y = 198.27

# Every raw Statcast column the pipeline reads. spray_angle_pull is derived, not downloaded.
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {*VIDEO_FEATURES, *LEAK_FREE_FEATURES} - {"spray_angle_pull"}
    | {"game_type", "game_date", "game_pk", "events", "des", "batter", "stand", "hc_x", "hc_y"}
)


def load_statcast(
    start_date: str, end_date: str, cache_path: Path, refresh: bool = False
) -> pd.DataFrame:
    """Return every pitch between two dates, reading from a local cache when available.

    A full season is ~750k+ rows and can take 10-30 minutes to download, so the first
    pull is pickled to ``cache_path`` and reused on later runs.
    """
    if cache_path.exists() and not refresh:
        logger.info("Loading cached Statcast data from %s", cache_path)
        return pd.read_pickle(cache_path)

    # Imported lazily so tests and cached runs don't need a network-capable pybaseball.
    from pybaseball import cache, statcast

    cache.enable()
    if refresh and hasattr(cache, "purge"):
        # pybaseball keeps its own cache; clear it so late-arriving games are re-fetched.
        cache.purge()

    logger.info("Downloading Statcast data %s to %s (this can take a while)", start_date, end_date)
    pitches = statcast(start_dt=start_date, end_dt=end_date)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pitches.to_pickle(cache_path)
    logger.info("Saved %s pitches to %s", f"{len(pitches):,}", cache_path)
    return pitches


def validate_columns(pitches: pd.DataFrame) -> None:
    """Fail early with a clear message if Statcast renamed or dropped a needed column."""
    missing = sorted(REQUIRED_COLUMNS - set(pitches.columns))
    if missing:
        raise ValueError(
            f"Statcast data is missing required columns: {missing}. Baseball Savant may have "
            "renamed them this season; update the feature lists in config.py to match."
        )


def coverage_report(pitches: pd.DataFrame, expected_end: str) -> dict[str, object]:
    """Summarize which regular-season dates and games the download actually contains.

    Statcast can take a day or so to post the final games, so this warns when the data
    stops short of the requested end date.
    """
    regular = pitches.loc[pitches["game_type"].eq("R")]
    dates = pd.to_datetime(regular["game_date"])
    report = {
        "first_game_date": str(dates.min().date()) if len(dates) else None,
        "last_game_date": str(dates.max().date()) if len(dates) else None,
        "regular_season_games": int(regular["game_pk"].nunique()),
        "pitches": int(len(regular)),
    }
    logger.info(
        "Coverage: %s to %s, %s games, %s pitches",
        report["first_game_date"],
        report["last_game_date"],
        f"{report['regular_season_games']:,}",
        f"{report['pitches']:,}",
    )
    if report["last_game_date"] and report["last_game_date"] < expected_end:
        logger.warning(
            "Data ends on %s but you asked for games through %s. Statcast may not have posted "
            "the final games yet; re-run later with --refresh-data.",
            report["last_game_date"],
            expected_end,
        )
    return report


def prepare_batted_balls(pitches: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season, non-bunt balls in play and add the binary ``hit`` target."""
    in_play = pitches["game_type"].eq("R") & pitches["events"].isin(EVENTS_TO_KEEP)
    bip = pitches.loc[in_play]

    # The events filter removes sac bunts, but bunt singles/groundouts slip through.
    is_bunt = bip["des"].str.contains(BUNT_PATTERN, case=False, na=False, regex=True)
    bip = bip.loc[~is_bunt].copy()

    bip["hit"] = bip["events"].isin(HIT_EVENTS).astype(int)
    logger.info(
        "Batted balls: %s (removed %s bunts), hit rate %.3f",
        f"{len(bip):,}",
        f"{int(is_bunt.sum()):,}",
        bip["hit"].mean(),
    )
    return bip


def add_spray_angle(bip: pd.DataFrame) -> pd.DataFrame:
    """Add ``spray_angle_pull``: horizontal direction in degrees, positive = pulled.

    Flipped by batter handedness so a right-handed batter's ball to left field and a
    left-handed batter's ball to right field are both positive.
    """
    out = bip.copy()
    raw = np.degrees(np.arctan2(out["hc_x"] - HOME_PLATE_X, HOME_PLATE_Y - out["hc_y"]))
    out["spray_angle_pull"] = np.select(
        [out["stand"].eq("R"), out["stand"].eq("L")], [-raw, raw], default=np.nan
    )
    return out
