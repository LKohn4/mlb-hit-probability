"""Shared scoring so every model is judged the same way."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from hit_probability.config import RANDOM_STATE
from hit_probability.features import SplitData


def score(
    model_name: str, method: str, y_true: np.ndarray, p_hit: np.ndarray, train_time_s: float
) -> dict[str, float | str]:
    """Accuracy at a 0.5 threshold plus probability-quality metrics."""
    p = np.clip(p_hit, 1e-7, 1 - 1e-7)
    return {
        "model": model_name,
        "method": method,
        "accuracy": accuracy_score(y_true, p >= 0.5),
        "roc_auc": roc_auc_score(y_true, p),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
        "train_time_s": train_time_s,
    }


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    """Accuracy of always predicting the most common class (usually 'out')."""
    rate = float(np.mean(y_true))
    return max(rate, 1 - rate)


def impurity_importance(model, feature_names) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def permutation_importance_table(model, split: SplitData, n_samples: int = 5000) -> pd.Series:
    """Drop in accuracy when each feature is shuffled, on a test-set subsample."""
    rng = np.random.default_rng(RANDOM_STATE)
    pos = rng.choice(len(split.X_test), size=min(n_samples, len(split.X_test)), replace=False)
    result = permutation_importance(
        model,
        split.X_test.iloc[pos],
        split.y_test[pos],
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=split.X_test.columns).sort_values(
        ascending=False
    )


def to_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table (no extra dependencies)."""

    def fmt(value) -> str:
        return float_fmt.format(value) if isinstance(value, float) else str(value)

    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])
