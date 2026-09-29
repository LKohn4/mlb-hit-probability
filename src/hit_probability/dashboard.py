"""Build every dashboard figure plus the CSV data behind each one.

Figures go to ``results/figures`` and the underlying numbers to ``results/dashboard_data``,
so you can post the PNGs directly or rebuild the charts in Tableau, Power BI, Canva, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hit_probability import plots
from hit_probability.config import CATEGORICAL_FEATURES, LEAK_FREE_FEATURES, Config
from hit_probability.data import HOME_PLATE_X, HOME_PLATE_Y
from hit_probability.expected import (
    add_names,
    hits_above_expected,
    lookup_names,
    out_of_fold_hit_probability,
)

logger = logging.getLogger(__name__)

HEATMAP_MIN_COUNT = 15
PITCH_MIN_BIP = 300


# --------------------------------------------------------------------------- tidy tables


def heatmap_grid(bip: pd.DataFrame, ev_step: int = 2, la_step: int = 4) -> pd.DataFrame:
    """Hit rate in exit-velocity x launch-angle cells, with bin edges as columns."""
    d = bip.dropna(subset=["launch_speed", "launch_angle"])
    ev_edges = np.arange(60, 120 + ev_step, ev_step)
    la_edges = np.arange(-40, 72 + la_step, la_step)
    ev_bin = pd.cut(d["launch_speed"], ev_edges, right=False, labels=ev_edges[:-1])
    la_bin = pd.cut(d["launch_angle"], la_edges, right=False, labels=la_edges[:-1])
    grid = (
        d.groupby([la_bin.rename("la_lo"), ev_bin.rename("ev_lo")], observed=False)["hit"]
        .agg(hit_rate="mean", balls_in_play="size")
        .reset_index()
    )
    grid["la_lo"] = grid["la_lo"].astype(float)
    grid["ev_lo"] = grid["ev_lo"].astype(float)
    grid["la_hi"] = grid["la_lo"] + la_step
    grid["ev_hi"] = grid["ev_lo"] + ev_step
    grid.loc[grid["balls_in_play"] < HEATMAP_MIN_COUNT, "hit_rate"] = np.nan
    return grid[["ev_lo", "ev_hi", "la_lo", "la_hi", "balls_in_play", "hit_rate"]]


def rate_curve(bip: pd.DataFrame, column: str, lo: float, hi: float, step: float) -> pd.DataFrame:
    """Hit rate and share of batted balls across equal-width bins of one column."""
    d = bip.dropna(subset=[column])
    edges = np.arange(lo, hi + step, step)
    binned = pd.cut(d[column], edges, right=False, labels=edges[:-1] + step / 2)
    curve = (
        d.groupby(binned.rename("bin_mid"), observed=False)["hit"]
        .agg(hit_rate="mean", balls_in_play="size")
        .reset_index()
    )
    curve["bin_mid"] = curve["bin_mid"].astype(float)
    curve["share"] = curve["balls_in_play"] / len(d)
    return curve.loc[curve["balls_in_play"] >= 30].reset_index(drop=True)


def pitch_type_table(bip: pd.DataFrame, min_bip: int) -> pd.DataFrame:
    table = (
        bip.dropna(subset=["pitch_type"])
        .groupby("pitch_type")["hit"]
        .agg(hit_rate="mean", balls_in_play="size")
        .query("balls_in_play >= @min_bip")
        .reset_index()
    )
    table["pitch_name"] = table["pitch_type"].map(plots.PITCH_NAMES).fillna(table["pitch_type"])
    return table


def spray_points(bip: pd.DataFrame) -> pd.DataFrame:
    d = bip.dropna(subset=["hc_x", "hc_y"])
    return pd.DataFrame(
        {"x": d["hc_x"] - HOME_PLATE_X, "y": HOME_PLATE_Y - d["hc_y"], "hit": d["hit"]}
    )


# --------------------------------------------------------------------------- build


def build_dashboard(
    bip: pd.DataFrame,
    comparison: pd.DataFrame,
    baseline: float,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    importance: pd.Series,
    nn_history: dict[str, list[float]] | None,
    coverage: dict,
    config: Config,
) -> list[Path]:
    fig_dir, data_dir = config.figures_dir, config.dashboard_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    season, credit = config.season_label, config.credit
    league_rate = float(bip["hit"].mean())
    made: list[Path] = []

    # Headline numbers
    # Headline the leak-free model when available: its score reflects contact quality alone,
    # while the video feature set is inflated by after-the-fact information.
    leak_free = comparison[comparison["method"].str.contains("leak-free")]
    pool = leak_free if len(leak_free) else comparison
    best = pool.loc[pool["roc_auc"].idxmax()]
    auc_label = "Model ROC AUC" if len(leak_free) else "Best ROC AUC"
    kpis = [
        (f"{len(bip):,}", "Batted balls", "regular season, bunts excluded"),
        (f"{coverage.get('regular_season_games', 0):,}", "Games",
         f"{coverage.get('first_game_date')} to {coverage.get('last_game_date')}"),
        (plots.avg_fmt(league_rate), "League hit rate", "on balls in play"),
        (f"{best['roc_auc']:.3f}", auc_label, f"{best['model']}, {best['method']}"),
    ]  # fmt: skip
    made.append(plots.season_summary_card(kpis, fig_dir / "season_summary.png", season, credit))
    (data_dir / "season_summary.json").write_text(
        json.dumps({"league_hit_rate": league_rate, "batted_balls": len(bip),
                    "majority_baseline_accuracy": baseline, **coverage}, indent=2),
        encoding="utf-8",
    )  # fmt: skip

    # Batted-ball physics
    grid = heatmap_grid(bip)
    grid.to_csv(data_dir / "hit_rate_heatmap.csv", index=False)
    made.append(plots.hit_rate_heatmap(grid, fig_dir / "hit_rate_heatmap.png", season, credit,
                                       HEATMAP_MIN_COUNT))  # fmt: skip

    la_curve = rate_curve(bip, "launch_angle", -40, 70, 5)
    la_curve.to_csv(data_dir / "hit_rate_by_launch_angle.csv", index=False)
    made.append(plots.hit_rate_curve(
        la_curve, league_rate, fig_dir / "hit_rate_by_launch_angle.png",
        title="The Launch Angle Sweet Spot",
        subtitle=f"Hit rate on balls in play by launch angle, {season}. Bars show how often "
                 "each angle happens.",
        xlabel="Launch angle (°)", credit=credit, color=plots.THEME["orange"],
    ))  # fmt: skip

    ev_curve = rate_curve(bip, "launch_speed", 50, 120, 2)
    ev_curve.to_csv(data_dir / "hit_rate_by_exit_velocity.csv", index=False)
    made.append(plots.hit_rate_curve(
        ev_curve, league_rate, fig_dir / "hit_rate_by_exit_velocity.png",
        title="Hit It Hard",
        subtitle=f"Hit rate on balls in play by exit velocity, {season}. Bars show how often "
                 "each speed happens.",
        xlabel="Exit velocity (mph)", credit=credit, color=plots.THEME["blue"],
    ))  # fmt: skip

    made.append(plots.spray_chart(spray_points(bip), fig_dir / "spray_chart.png", season, credit))

    pitch_min = max(20, int(PITCH_MIN_BIP * (config.sample_frac or 1)))
    pitches = pitch_type_table(bip, pitch_min)
    pitches.to_csv(data_dir / "hit_rate_by_pitch_type.csv", index=False)
    if len(pitches):
        made.append(plots.hit_rate_by_pitch_type(
            pitches, league_rate, fig_dir / "hit_rate_by_pitch_type.png", season, credit))

    # Model evaluation
    made.append(plots.model_comparison(comparison, baseline, fig_dir / "model_comparison.png",
                                       season, credit))  # fmt: skip
    made.append(plots.calibration_chart(curves, fig_dir / "calibration.png", season, credit))
    made.append(plots.roc_chart(curves, fig_dir / "roc_curves.png", season, credit))

    importance.rename("accuracy_drop").rename_axis("feature").reset_index().assign(
        label=lambda t: t["feature"].map(plots.pretty_feature)
    ).to_csv(data_dir / "feature_importance.csv", index=False)
    made.append(plots.feature_importance(importance, fig_dir / "feature_importance.png",
                                         season, credit))  # fmt: skip
    if nn_history:
        made.append(plots.training_curves(nn_history, fig_dir / "nn_training_curves.png",
                                          season, credit))  # fmt: skip

    # Player leaderboard
    made += _leaderboard(bip, config)

    logger.info("Dashboard: %d figures in %s", len(made), fig_dir.resolve())
    return made


def _leaderboard(bip: pd.DataFrame, config: Config) -> list[Path]:
    scored = out_of_fold_hit_probability(
        bip, LEAK_FREE_FEATURES, CATEGORICAL_FEATURES, config.n_estimators, config.xh_folds
    )
    min_bip = max(10, int(config.min_bip_leaderboard * (config.sample_frac or 1)))
    board = hits_above_expected(scored, min_bip)
    if board.empty:
        logger.warning("No batters reached %d balls in play; skipping the leaderboard.", min_bip)
        return []

    names = lookup_names(board["batter"]) if config.lookup_names else {}
    board = add_names(board, names)
    board.round(4).to_csv(config.dashboard_data_dir / "hits_above_expected.csv", index=False)
    return [plots.hits_above_expected_chart(board, config.figures_dir / "hits_above_expected.png",
                                            config.season_label, config.credit, min_bip)]
