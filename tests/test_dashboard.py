import pandas as pd

from hit_probability.dashboard import heatmap_grid, pitch_type_table, rate_curve
from hit_probability.expected import add_names, hits_above_expected
from hit_probability.plots import avg_fmt, pretty_feature


def test_heatmap_grid_has_edges_and_masks_sparse_cells(batted_balls):
    grid = heatmap_grid(batted_balls)
    assert {"ev_lo", "ev_hi", "la_lo", "la_hi", "hit_rate"} <= set(grid.columns)
    assert (grid["ev_hi"] - grid["ev_lo"] == 2).all()
    assert grid.loc[grid["balls_in_play"] < 15, "hit_rate"].isna().all()


def test_rate_curve_shares_are_valid(batted_balls):
    curve = rate_curve(batted_balls, "launch_angle", -40, 70, 5)
    assert curve["hit_rate"].between(0, 1).all()
    assert curve["share"].sum() <= 1.0 + 1e-9


def test_pitch_type_table_names_pitches(batted_balls):
    table = pitch_type_table(batted_balls, min_bip=10)
    assert "4-Seam Fastball" in set(table["pitch_name"])


def test_hits_above_expected_math():
    scored = pd.DataFrame(
        {"batter": [1, 1, 1, 2, 2, 2], "hit": [1, 1, 0, 0, 0, 0],
         "x_hit": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
    )
    board = hits_above_expected(scored, min_bip=3)
    assert list(board["batter"]) == [1, 2]
    assert board["hits_above_expected"].tolist() == [0.5, -1.5]
    named = add_names(board, {1: "Test Player"})
    assert named["name"].tolist() == ["Test Player", "ID 2"]


def test_labels():
    assert pretty_feature("pitch_type_ST") == "Pitch: Sweeper"
    assert pretty_feature("if_fielding_alignment_Strategic") == "IF alignment: Strategic"
    assert pretty_feature("launch_speed") == "Exit velocity"
    assert avg_fmt(0.3264) == ".326"
