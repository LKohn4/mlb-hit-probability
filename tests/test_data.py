import numpy as np
import pandas as pd

from hit_probability.config import EVENTS_TO_KEEP, HIT_EVENTS
from hit_probability.data import add_spray_angle, load_statcast, prepare_batted_balls


def test_keeps_only_regular_season_batted_balls(fake_pitches):
    bip = prepare_batted_balls(fake_pitches)
    assert (bip["game_type"] == "R").all()
    assert bip["events"].isin(EVENTS_TO_KEEP).all()


def test_removes_bunts(fake_pitches):
    bip = prepare_batted_balls(fake_pitches)
    assert not bip["des"].str.contains("bunt", case=False).any()


def test_hit_target_matches_events(fake_pitches):
    bip = prepare_batted_balls(fake_pitches)
    assert set(bip["hit"].unique()) <= {0, 1}
    assert (bip["hit"] == bip["events"].isin(HIT_EVENTS)).all()


def test_spray_angle_positive_means_pulled():
    df = pd.DataFrame({"hc_x": [60.0, 60.0, 190.0], "hc_y": [100.0, 100.0, 100.0],
                       "stand": ["R", "L", None]})
    angles = add_spray_angle(df)["spray_angle_pull"]
    assert angles.iloc[0] > 0  # righty to left field = pulled
    assert angles.iloc[1] < 0  # lefty to left field = opposite field
    assert np.isnan(angles.iloc[2])  # unknown handedness


def test_load_statcast_uses_cache(tmp_path, fake_pitches):
    cache_path = tmp_path / "cached.pkl"
    fake_pitches.to_pickle(cache_path)
    loaded = load_statcast("2025-03-18", "2025-09-28", cache_path)
    pd.testing.assert_frame_equal(loaded, fake_pitches)


def test_validate_columns_reports_missing(fake_pitches):
    import pytest

    from hit_probability.data import validate_columns

    validate_columns(fake_pitches)  # complete data passes
    with pytest.raises(ValueError, match="bat_speed"):
        validate_columns(fake_pitches.drop(columns=["bat_speed"]))


def test_coverage_report_warns_when_data_ends_early(fake_pitches, caplog):
    from hit_probability.data import coverage_report

    report = coverage_report(fake_pitches, expected_end="2026-12-31")
    assert report["first_game_date"] >= "2026-03-25"
    assert report["regular_season_games"] > 0
    assert "Statcast may not have posted" in caplog.text
