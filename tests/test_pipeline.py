"""Smoke test of the full pipeline on cached synthetic data."""

from hit_probability.config import Config
from hit_probability.pipeline import run

from .conftest import make_fake_pitches


def test_pipeline_writes_results(tmp_path):
    config = Config(data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                    n_estimators=10, run_nn=False, permutation_sample=500,
                    lookup_names=False, min_bip_leaderboard=20, credit="@test")
    config.data_dir.mkdir()
    make_fake_pitches().to_pickle(config.cache_path)

    comparison = run(config)

    assert len(comparison) == 3  # RF x (strict, imputed, leak-free)
    for name in ("model_comparison.csv", "model_comparison.md", "summary.json",
                 "missingness.csv", "outcome_survival.csv"):
        assert (config.results_dir / name).exists(), name

    figures = {p.name for p in config.figures_dir.glob("*.png")}
    expected = {
        "season_summary.png", "hit_rate_heatmap.png", "hit_rate_by_launch_angle.png",
        "hit_rate_by_exit_velocity.png", "spray_chart.png", "hit_rate_by_pitch_type.png",
        "model_comparison.png", "calibration.png", "roc_curves.png",
        "feature_importance.png", "hits_above_expected.png",
    }
    assert expected <= figures, expected - figures
    assert (config.dashboard_data_dir / "hits_above_expected.csv").exists()
