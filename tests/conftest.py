"""Synthetic Statcast-shaped data so tests run offline in seconds."""

import numpy as np
import pandas as pd
import pytest


def make_fake_pitches(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    events = rng.choice(
        ["field_out", "single", "double", "home_run", "force_out", "strikeout", "ball",
         "sac_bunt", "field_error", "triple"],
        n,
        p=[0.35, 0.12, 0.05, 0.03, 0.03, 0.15, 0.20, 0.02, 0.02, 0.03],
    )
    df = pd.DataFrame(
        {
            "game_type": rng.choice(["R", "S"], n, p=[0.95, 0.05]),
            "events": events,
            "des": np.where(
                rng.random(n) < 0.03, "Smith singles on a bunt ground ball", "Smith lines out"
            ),
            "launch_speed": rng.normal(88, 14, n),
            "launch_angle": rng.normal(12, 25, n),
            "pitch_type": rng.choice(["FF", "SL", "CH", "CU", "SI"], n),
            # Home runs have no fielder, so no hit_location (mirrors real Statcast).
            "hit_location": np.where(
                events == "home_run", np.nan, rng.integers(1, 10, n).astype(float)
            ),
            "bat_speed": np.where(rng.random(n) < 0.1, np.nan, rng.normal(70, 6, n)),
            "swing_length": rng.normal(7.3, 0.8, n),
            "swing_path_tilt": rng.normal(30, 5, n),
            "if_fielding_alignment": rng.choice(["Standard", "Strategic", "Infield shade"], n),
            "of_fielding_alignment": rng.choice(["Standard", "Strategic"], n),
            "hit_distance_sc": rng.normal(170, 110, n),
            "release_speed": rng.normal(89, 6, n),
            "hc_x": rng.normal(125, 40, n),
            "hc_y": rng.normal(130, 40, n),
            "stand": rng.choice(["R", "L"], n),
            "batter": rng.integers(600_000, 600_030, n),
            "game_date": pd.Timestamp("2026-03-25") + pd.to_timedelta(rng.integers(0, 187, n), "D"),
            "game_pk": rng.integers(1, 2_430, n),
        }
    )
    df.loc[rng.random(n) < 0.03, "pitch_type"] = None
    return df


@pytest.fixture
def fake_pitches() -> pd.DataFrame:
    return make_fake_pitches()


@pytest.fixture
def batted_balls(fake_pitches):
    from hit_probability.data import add_spray_angle, prepare_batted_balls

    return add_spray_angle(prepare_batted_balls(fake_pitches))
