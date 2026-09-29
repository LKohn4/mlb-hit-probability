"""Train/test preparation for the two missing-data strategies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hit_probability.config import RANDOM_STATE


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def n_rows(self) -> int:
        return len(self.X_train) + len(self.X_test)


def _split(X: pd.DataFrame, y: np.ndarray, test_size: float):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def prepare_strict(
    bip: pd.DataFrame,
    features: Sequence[str],
    categorical: Sequence[str],
    test_size: float = 0.2,
) -> SplitData:
    """Method A (the video's approach): keep only rows with every feature present."""
    features, categorical = list(features), [c for c in categorical if c in features]
    complete = bip.dropna(subset=features)
    X = pd.get_dummies(complete[features], columns=categorical, dtype=float)
    y = complete["hit"].to_numpy()
    return SplitData(*_split(X, y, test_size))


def prepare_imputed(
    bip: pd.DataFrame,
    features: Sequence[str],
    categorical: Sequence[str],
    test_size: float = 0.2,
) -> SplitData:
    """Method B: fill numeric gaps with the median and categorical gaps with the mode.

    Fill values are learned from the training split only, so the test set never
    influences them.
    """
    features, categorical = list(features), [c for c in categorical if c in features]
    numeric = [c for c in features if c not in categorical]

    X_train, X_test, y_train, y_test = _split(bip[features], bip["hit"].to_numpy(), test_size)

    fill_values = pd.concat([X_train[numeric].median(), X_train[categorical].mode().iloc[0]])
    X_train, X_test = X_train.fillna(fill_values), X_test.fillna(fill_values)

    X_train = pd.get_dummies(X_train, columns=categorical, dtype=float)
    # A rare category may appear in only one split; align test columns to train.
    X_test = pd.get_dummies(X_test, columns=categorical, dtype=float).reindex(
        columns=X_train.columns, fill_value=0.0
    )
    return SplitData(X_train, X_test, y_train, y_test)


def missingness_report(bip: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    """Share of rows missing each feature, highest first."""
    return bip[list(features)].isna().mean().sort_values(ascending=False).rename("share_missing")


def outcome_survival(bip: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """How many rows of each outcome survive the strict ``dropna``.

    Useful for spotting outcomes (e.g. home runs, which have no fielder and therefore no
    ``hit_location``) that the strict method silently removes.
    """
    kept = bip.dropna(subset=list(features))
    counts = {
        "before": bip["events"].value_counts(),
        "after_strict_dropna": kept["events"].value_counts(),
    }
    table = pd.DataFrame(counts).fillna(0).astype(int)
    table["kept_pct"] = (100 * table["after_strict_dropna"] / table["before"]).round(1)
    return table.sort_values("before", ascending=False)
