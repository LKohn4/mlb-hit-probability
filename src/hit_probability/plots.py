"""Dashboard-ready figures with one consistent dark theme.

Every figure has a title, subtitle and a source/credit footer so it can be posted on its own
or tiled into a dashboard. Plotting functions take tidy DataFrames (built in ``dashboard.py``)
so the same numbers can be exported to CSV for other dashboard tools.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; works headless and in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

THEME = {
    "bg": "#0E1117",
    "panel": "#161B22",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "grid": "#30363D",
    "orange": "#F97316",
    "blue": "#38BDF8",
    "green": "#22C55E",
    "red": "#EF4444",
    "yellow": "#FACC15",
}
SERIES_COLORS = ["#F97316", "#38BDF8", "#A78BFA", "#22C55E", "#FACC15", "#F472B6"]
HIT_CMAP = LinearSegmentedColormap.from_list(
    "hit_rate", ["#1E3A8A", "#2563EB", "#38BDF8", "#FDE047", "#F97316", "#DC2626"]
)
SOURCE = "Data: MLB Statcast via Baseball Savant & pybaseball"
FIGSIZE = (10, 6.25)  # 2000 x 1250 px at 200 dpi
DPI = 200

PITCH_NAMES = {
    "FF": "4-Seam Fastball", "SI": "Sinker", "FC": "Cutter", "FA": "Fastball",
    "SL": "Slider", "ST": "Sweeper", "SV": "Slurve", "CU": "Curveball",
    "KC": "Knuckle Curve", "CS": "Slow Curve", "CH": "Changeup", "FS": "Splitter",
    "FO": "Forkball", "SC": "Screwball", "KN": "Knuckleball", "EP": "Eephus",
}  # fmt: skip
FEATURE_NAMES = {
    "launch_speed": "Exit velocity",
    "launch_angle": "Launch angle",
    "spray_angle_pull": "Spray angle (pull +)",
    "hit_location": "Hit location (fielder)",
    "hit_distance_sc": "Hit distance",
    "bat_speed": "Bat speed",
    "swing_length": "Swing length",
    "swing_path_tilt": "Swing path tilt",
    "release_speed": "Pitch velocity",
}
_PREFIXES = {
    "pitch_type_": "Pitch: ",
    "if_fielding_alignment_": "IF alignment: ",
    "of_fielding_alignment_": "OF alignment: ",
}


def pretty_feature(name: str) -> str:
    """Human-readable label for a raw or one-hot-encoded feature name."""
    if name in FEATURE_NAMES:
        return FEATURE_NAMES[name]
    for prefix, label in _PREFIXES.items():
        if name.startswith(prefix):
            code = name[len(prefix) :]
            return label + PITCH_NAMES.get(code, code)
    return name


def avg_fmt(x: float) -> str:
    """Baseball-style rate: 0.3264 -> '.326'."""
    return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"


# --------------------------------------------------------------------------- helpers


@contextmanager
def _style():
    rc = {
        "figure.facecolor": THEME["bg"],
        "axes.facecolor": THEME["bg"],
        "savefig.facecolor": THEME["bg"],
        "axes.edgecolor": THEME["grid"],
        "axes.labelcolor": THEME["muted"],
        "axes.titlecolor": THEME["text"],
        "xtick.color": THEME["muted"],
        "ytick.color": THEME["muted"],
        "text.color": THEME["text"],
        "axes.grid": True,
        "grid.color": THEME["grid"],
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "legend.labelcolor": THEME["text"],
    }
    with plt.rc_context(rc):
        yield


def _figure(title: str, subtitle: str, credit: str, ncols: int = 1, figsize=FIGSIZE, **adjust):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    margins = {"top": 0.80, "bottom": 0.13, "left": 0.09, "right": 0.96, "wspace": 0.35}
    fig.subplots_adjust(**{**margins, **adjust})
    fig.text(0.04, 0.955, title, fontsize=20, fontweight="bold", va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=11.5, va="top", color=THEME["muted"], wrap=True)
    footer = SOURCE + (f"   |   {credit}" if credit else "")
    fig.text(0.04, 0.025, footer, fontsize=9, color=THEME["muted"])
    return fig, axes


def _colorbar(fig, mappable, ax, label: str):
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(label, color=THEME["muted"])
    cbar.ax.yaxis.set_tick_params(color=THEME["muted"])
    cbar.outline.set_edgecolor(THEME["grid"])
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: avg_fmt(v)))
    return cbar


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _styled(func: Callable) -> Callable:
    """Run a plotting function inside the dashboard theme."""

    def wrapper(*args, **kwargs):
        with _style():
            return func(*args, **kwargs)

    wrapper.__name__, wrapper.__doc__ = func.__name__, func.__doc__
    return wrapper


# --------------------------------------------------------------------------- figures


@_styled
def season_summary_card(kpis: list[tuple[str, str, str]], path: Path, season: str, credit: str):
    """Row of big-number tiles: (value, label, caption)."""
    fig, axes = _figure(
        "Hit Probability Model", season, credit, ncols=len(kpis), figsize=(12, 4.2),
        top=0.72, bottom=0.14, left=0.03, right=0.97, wspace=0.08,
    )  # fmt: skip
    for ax, (value, label, caption), color in zip(axes, kpis, SERIES_COLORS, strict=False):
        ax.set_axis_off()
        ax.add_patch(
            FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.06",
                           transform=ax.transAxes, facecolor=THEME["panel"],
                           edgecolor=THEME["grid"])
        )  # fmt: skip
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=30, fontweight="bold",
                color=color, transform=ax.transAxes)  # fmt: skip
        ax.text(0.5, 0.33, label, ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.16, caption, ha="center", va="center", fontsize=9,
                color=THEME["muted"], transform=ax.transAxes)  # fmt: skip
    return _save(fig, path)


@_styled
def hit_rate_heatmap(grid: pd.DataFrame, path: Path, season: str, credit: str, min_count: int):
    """Hit rate by exit velocity x launch angle (tidy grid with *_lo/*_hi bin edges)."""
    ev_edges = np.unique(np.r_[grid["ev_lo"], grid["ev_hi"]])
    la_edges = np.unique(np.r_[grid["la_lo"], grid["la_hi"]])
    rate = (
        grid.pivot(index="la_lo", columns="ev_lo", values="hit_rate")
        .reindex(index=la_edges[:-1], columns=ev_edges[:-1])
        .to_numpy()
    )

    fig, ax = _figure(
        "Where Hits Come From",
        f"Hit rate on balls in play by exit velocity and launch angle, {season} "
        f"(cells with {min_count}+ batted balls)",
        credit,
    )
    mesh = ax.pcolormesh(ev_edges, la_edges, np.ma.masked_invalid(rate), cmap=HIT_CMAP,
                         vmin=0, vmax=1, shading="flat")  # fmt: skip
    ax.grid(False)
    ax.set_xlabel("Exit velocity (mph)")
    ax.set_ylabel("Launch angle (°)")
    ax.axhline(0, color=THEME["muted"], lw=0.8, ls=":")
    _colorbar(fig, mesh, ax, "Hit rate")
    return _save(fig, path)


@_styled
def hit_rate_curve(
    curve: pd.DataFrame, league_rate: float, path: Path, *, title: str, subtitle: str,
    xlabel: str, credit: str, color: str,
):  # fmt: skip
    """Hit rate (line) and share of batted balls (bars) across bins of one variable."""
    fig, ax = _figure(title, subtitle, credit)
    width = float(np.median(np.diff(curve["bin_mid"]))) if len(curve) > 1 else 1.0

    ax_vol = ax.twinx()
    ax_vol.bar(curve["bin_mid"], curve["share"], width=width * 0.9, color=THEME["muted"],
               alpha=0.18, zorder=0)  # fmt: skip
    ax_vol.set_ylabel("Share of batted balls")
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_vol.grid(False)
    ax_vol.spines["right"].set_visible(True)
    ax.set_zorder(ax_vol.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot(curve["bin_mid"], curve["hit_rate"], color=color, lw=2.5, marker="o", ms=4)
    ax.axhline(league_rate, color=THEME["muted"], ls="--", lw=1)
    ax.text(curve["bin_mid"].min(), league_rate, f"  League avg {avg_fmt(league_rate)}",
            va="bottom", fontsize=9, color=THEME["muted"])  # fmt: skip
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Hit rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: avg_fmt(v)))
    return _save(fig, path)


@_styled
def spray_chart(points: pd.DataFrame, path: Path, season: str, credit: str):
    """Hexbin hit-rate map over the field. ``points`` needs x, y (feet-ish units) and hit."""
    fig, ax = _figure(
        "Spray Chart: Hit Rate by Field Location",
        f"Where batted balls were fielded or landed, colored by hit rate, {season}",
        credit, left=0.05, right=0.93, bottom=0.08,
    )  # fmt: skip
    hb = ax.hexbin(points["x"], points["y"], C=points["hit"], reduce_C_function=np.mean,
                   gridsize=45, mincnt=10, cmap=HIT_CMAP, vmin=0, vmax=1,
                   extent=(-120, 120, -6, 176), linewidths=0.1)  # fmt: skip

    # Approximate field outline in Statcast hit-coordinate units (~2.5 ft per unit).
    theta = np.radians(np.linspace(45, 135, 200))
    fence = 132 + 28 * np.sin(2 * (theta - np.radians(45)))
    ax.plot(fence * np.cos(theta), fence * np.sin(theta), color=THEME["text"], lw=1.2)
    ax.plot(62 * np.cos(theta), 62 * np.sin(theta), color=THEME["muted"], lw=0.8, ls="--")
    for sign in (-1, 1):
        ax.plot([0, sign * 132 * np.cos(np.radians(45))], [0, 132 * np.sin(np.radians(45))],
                color=THEME["text"], lw=1.2)  # fmt: skip
    for label, x, y in (("LF", -102, 112), ("CF", 0, 166), ("RF", 102, 112)):
        ax.text(x, y, label, ha="center", fontsize=12, fontweight="bold", color=THEME["muted"])

    ax.set_xlim(-120, 120)
    ax.set_ylim(-6, 176)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _colorbar(fig, hb, ax, "Hit rate")
    return _save(fig, path)


@_styled
def hit_rate_by_pitch_type(table: pd.DataFrame, league_rate: float, path: Path, season: str,
                           credit: str):  # fmt: skip
    """Horizontal bars of hit rate on balls in play for each pitch type."""
    table = table.sort_values("hit_rate")
    fig, ax = _figure(
        "Which Pitches Get Hit (When Put in Play)",
        f"Hit rate on balls in play by pitch type, {season}", credit, left=0.22,
    )  # fmt: skip
    colors = [THEME["orange"] if r >= league_rate else THEME["blue"] for r in table["hit_rate"]]
    ax.barh(table["pitch_name"], table["hit_rate"], color=colors)
    for y, (rate, n) in enumerate(zip(table["hit_rate"], table["balls_in_play"], strict=True)):
        ax.text(rate + 0.004, y, f"{avg_fmt(rate)}  ({n:,} BIP)", va="center", fontsize=9)
    ax.axvline(league_rate, color=THEME["muted"], ls="--", lw=1)
    ax.set_xlim(0, table["hit_rate"].max() * 1.25)
    ax.set_xlabel("Hit rate on balls in play")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: avg_fmt(v)))
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


@_styled
def model_comparison(comparison: pd.DataFrame, baseline: float, path: Path, season: str,
                     credit: str):  # fmt: skip
    """Accuracy and ROC AUC for every model/method, with reference lines."""
    labels = [f"{m}\n{meth}" for m, meth in zip(comparison["model"], comparison["method"],
                                                  strict=True)]  # fmt: skip
    colors = [THEME["orange"] if m == "Random forest" else THEME["blue"]
              for m in comparison["model"]]  # fmt: skip
    fig, (ax_acc, ax_auc) = _figure(
        "Model Scorecard",
        f"Held-out test performance, {season}. Orange = random forest, blue = neural network.",
        credit, ncols=2, left=0.19, wspace=0.08,
    )  # fmt: skip
    y = np.arange(len(labels))[::-1]
    for ax, col, ref, ref_label in (
        (ax_acc, "accuracy", baseline, f"Always 'out': {baseline:.1%}"),
        (ax_auc, "roc_auc", 0.5, "Coin flip: 0.50"),
    ):
        ax.barh(y, comparison[col], color=colors, height=0.65)
        for yi, v in zip(y, comparison[col], strict=True):
            ax.text(v - 0.005, yi, f"{v:.3f}", va="center", ha="right", fontsize=9,
                    fontweight="bold", color=THEME["bg"])  # fmt: skip
        ax.axvline(ref, color=THEME["text"], ls="--", lw=1)
        ax.text(ref, -0.95, f" {ref_label}", fontsize=9, ha="left", va="center",
                color=THEME["muted"])  # fmt: skip
        ax.set_ylim(-1.3, len(labels) - 0.4)
        ax.set_xlim(min(ref, comparison[col].min()) - 0.05, 1.0)
        ax.set_title("Accuracy" if col == "accuracy" else "ROC AUC", loc="left")
        ax.grid(axis="y", visible=False)
    ax_acc.set_yticks(y, labels, fontsize=9)
    ax_auc.set_yticks(y, [""] * len(labels))
    return _save(fig, path)


@_styled
def calibration_chart(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path,
                      season: str, credit: str):  # fmt: skip
    """Predicted vs. actual hit rate: points on the diagonal mean trustworthy probabilities."""
    fig, ax = _figure(
        "Can You Trust the Probabilities?",
        f"Predicted vs. actual hit rate in 10 equal-size groups of test balls, {season}",
        credit, left=0.1, right=0.72,
    )  # fmt: skip
    ax.plot([0, 1], [0, 1], color=THEME["muted"], ls="--", lw=1, label="Perfect calibration")
    for (label, (y_true, p)), color in zip(curves.items(), SERIES_COLORS, strict=False):
        frac, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=4, lw=2, color=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted hit probability")
    ax.set_ylabel("Actual hit rate")
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=9)
    return _save(fig, path)


@_styled
def roc_chart(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, season: str,
              credit: str):  # fmt: skip
    fig, ax = _figure(
        "ROC Curves",
        f"How well each model separates hits from outs at every threshold, {season}",
        credit, left=0.1, right=0.72,
    )  # fmt: skip
    ax.plot([0, 1], [0, 1], color=THEME["muted"], ls="--", lw=1, label="Coin flip (AUC 0.50)")
    for (label, (y_true, p)), color in zip(curves.items(), SERIES_COLORS, strict=False):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{label} (AUC {roc_auc_score(y_true, p):.3f})")  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (outs called hits)")
    ax.set_ylabel("True positive rate (hits caught)")
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=9)
    return _save(fig, path)


@_styled
def feature_importance(importance: pd.Series, path: Path, season: str, credit: str,
                       top_n: int = 12):  # fmt: skip
    """Permutation importance with readable feature names."""
    top = importance.head(top_n).iloc[::-1]
    fig, ax = _figure(
        "What the Model Relies On",
        f"Drop in random-forest accuracy when each feature is shuffled, {season}",
        credit, left=0.28,
    )  # fmt: skip
    ax.barh([pretty_feature(n) for n in top.index], top.to_numpy(), color=THEME["orange"])
    for yi, v in enumerate(top.to_numpy()):
        ax.text(v, yi, f"  {v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, top.max() * 1.18 if top.max() > 0 else 1)
    ax.set_xlabel("Accuracy drop when shuffled")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


@_styled
def training_curves(history: dict[str, list[float]], path: Path, season: str, credit: str):
    fig, ax = _figure(
        "Neural Network Training",
        f"Accuracy on training data vs. held-back validation data by epoch, {season}", credit,
    )  # fmt: skip
    epochs = np.arange(1, len(history["accuracy"]) + 1)
    ax.plot(epochs, history["accuracy"], lw=2.5, color=THEME["orange"], label="Training")
    ax.plot(epochs, history["val_accuracy"], lw=2.5, color=THEME["blue"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.legend()
    return _save(fig, path)


@_styled
def hits_above_expected_chart(board: pd.DataFrame, path: Path, season: str, credit: str,
                              min_bip: int, top_n: int = 10):  # fmt: skip
    """Diverging bars: the luckiest and unluckiest hitters on balls in play."""
    top = board.head(top_n)
    bottom = board.tail(top_n) if len(board) > top_n else board.iloc[0:0]
    shown = pd.concat([top, bottom]).drop_duplicates("batter").sort_values("hits_above_expected")

    fig, ax = _figure(
        "Luckiest & Unluckiest Hitters",
        f"Actual hits minus expected hits on balls in play, {season} (min {min_bip} BIP). "
        "Expected hits use contact quality only, so fast runners tend to rank high.",
        credit, left=0.26, top=0.78,
    )  # fmt: skip
    colors = [THEME["green"] if v >= 0 else THEME["red"] for v in shown["hits_above_expected"]]
    labels = [f"{n}  ({h:.0f} H / {x:.1f} xH)" for n, h, x in
              zip(shown["name"], shown["hits"], shown["expected_hits"], strict=True)]  # fmt: skip
    ax.barh(labels, shown["hits_above_expected"], color=colors)
    ax.axvline(0, color=THEME["text"], lw=1)
    for yi, v in enumerate(shown["hits_above_expected"]):
        ax.text(v, yi, f" {v:+.1f} " if v >= 0 else f" {v:+.1f} ", va="center", fontsize=9,
                ha="left" if v >= 0 else "right")  # fmt: skip
    lim = shown["hits_above_expected"].abs().max() * 1.25 if len(shown) else 1
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Hits above expected")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
