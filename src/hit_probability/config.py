"""Project-wide constants and the run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RANDOM_STATE = 42


def _project_root() -> Path:
    """The folder holding this project's pyproject.toml, so data and results always land in
    the project no matter which working directory the program is started from. Falls back
    to the current directory for non-editable installs."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").exists() else Path.cwd()


PROJECT_ROOT = _project_root()

# Batted-ball outcomes kept for modeling. Sac bunts are excluded here; other bunts
# are removed separately in ``data.prepare_batted_balls``.
EVENTS_TO_KEEP: tuple[str, ...] = (
    "field_out",
    "force_out",
    "single",
    "double",
    "triple",
    "home_run",
    "field_error",
    "grounded_into_double_play",
    "sac_fly",
    "fielders_choice",
    "double_play",
    "fielders_choice_out",
    "sac_fly_double_play",
    "triple_play",
)
HIT_EVENTS: tuple[str, ...] = ("single", "double", "triple", "home_run")

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "pitch_type",
    "if_fielding_alignment",
    "of_fielding_alignment",
)

# The feature set used in the original video.
VIDEO_FEATURES: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "pitch_type",
    "hit_location",
    "bat_speed",
    "swing_length",
    "swing_path_tilt",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "hit_distance_sc",
    "release_speed",
)

# Drops features recorded after the ball is in play (hit_location, hit_distance_sc)
# and adds spray angle, derived in ``data.add_spray_angle``.
LEAK_FREE_FEATURES: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "spray_angle_pull",
    "bat_speed",
    "swing_length",
    "swing_path_tilt",
    "release_speed",
    "pitch_type",
    "if_fielding_alignment",
    "of_fielding_alignment",
)


@dataclass
class Config:
    """Settings for one pipeline run."""

    # 2026 regular season: Opening Night (Mar 25) through the final day (Sep 27).
    start_date: str = "2026-03-25"
    end_date: str = "2026-09-27"
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    refresh_data: bool = False
    sample_frac: float | None = None

    test_size: float = 0.2
    n_estimators: int = 100
    epochs: int = 20
    batch_size: int = 100
    permutation_sample: int = 5000

    run_nn: bool = True
    run_leak_free: bool = True

    # Dashboard figures
    make_dashboard: bool = True
    credit: str = ""  # e.g. "Leo Kohn", printed in every figure's footer
    lookup_names: bool = True  # look up batter names online for the leaderboard
    min_bip_leaderboard: int = 250  # minimum balls in play to appear on the leaderboard
    xh_folds: int = 5  # cross-validation folds for out-of-sample expected hits

    @property
    def cache_path(self) -> Path:
        return self.data_dir / f"statcast_{self.start_date}_{self.end_date}.pkl"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def dashboard_data_dir(self) -> Path:
        return self.results_dir / "dashboard_data"

    @property
    def season_label(self) -> str:
        start_year, end_year = self.start_date[:4], self.end_date[:4]
        years = start_year if start_year == end_year else f"{start_year}-{end_year}"
        return f"{years} MLB regular season"
