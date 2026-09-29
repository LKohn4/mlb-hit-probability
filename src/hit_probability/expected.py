"""Expected hits (xH) per batter and the hits-above-expected leaderboard.

Every batted ball gets an *out-of-sample* hit probability: the data is split into folds and
each fold is scored by a model that never saw it. Scoring balls with a model trained on them
would make predictions nearly match reality and hide over- and under-performance.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from hit_probability.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def out_of_fold_hit_probability(
    bip: pd.DataFrame,
    features: Sequence[str],
    categorical: Sequence[str],
    n_estimators: int = 100,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Return ``batter``, ``hit`` and out-of-sample ``x_hit`` for every complete batted ball."""
    features = list(features)
    categorical = [c for c in categorical if c in features]
    complete = bip.dropna(subset=features)

    X = pd.get_dummies(complete[features], columns=categorical, dtype=float)
    y = complete["hit"].to_numpy()

    # min_samples_leaf smooths the forest's probabilities; fully grown trees are overconfident.
    model = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    logger.info("Scoring %s batted balls out-of-sample (%d folds)", f"{len(X):,}", n_splits)
    p_hit = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]

    out = complete[["batter", "hit"]].copy()
    out["x_hit"] = p_hit
    return out


def hits_above_expected(scored: pd.DataFrame, min_bip: int) -> pd.DataFrame:
    """Aggregate to one row per batter, sorted from most to fewest hits above expected."""
    board = (
        scored.groupby("batter")
        .agg(balls_in_play=("hit", "size"), hits=("hit", "sum"), expected_hits=("x_hit", "sum"))
        .query("balls_in_play >= @min_bip")
    )
    board["hits_above_expected"] = board["hits"] - board["expected_hits"]
    board["babip_style_avg"] = board["hits"] / board["balls_in_play"]
    board["expected_avg"] = board["expected_hits"] / board["balls_in_play"]
    return board.sort_values("hits_above_expected", ascending=False).reset_index()


def lookup_names(mlbam_ids: Iterable[int]) -> dict[int, str]:
    """Map MLBAM batter IDs to "First Last" via pybaseball. Returns {} if the lookup fails."""
    ids = [int(i) for i in mlbam_ids]
    if not ids:
        return {}
    try:
        from pybaseball import playerid_reverse_lookup

        table = playerid_reverse_lookup(ids, key_type="mlbam")
        return {
            int(row.key_mlbam): f"{str(row.name_first).title()} {str(row.name_last).title()}"
            for row in table.itertuples()
        }
    except Exception as exc:  # network issues shouldn't sink the whole pipeline
        logger.warning("Player name lookup failed (%s); the leaderboard will show IDs.", exc)
        return {}


def add_names(board: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    out = board.copy()
    out.insert(1, "name", [names.get(int(b), f"ID {int(b)}") for b in out["batter"]])
    return out
