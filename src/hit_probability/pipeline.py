"""End-to-end experiment: data -> features -> models -> comparison table -> dashboard."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from hit_probability.config import (
    CATEGORICAL_FEATURES,
    LEAK_FREE_FEATURES,
    RANDOM_STATE,
    VIDEO_FEATURES,
    Config,
)
from hit_probability.data import (
    add_spray_angle,
    coverage_report,
    load_statcast,
    prepare_batted_balls,
    validate_columns,
)
from hit_probability.evaluate import (
    majority_baseline_accuracy,
    permutation_importance_table,
    score,
    to_markdown_table,
)
from hit_probability.features import (
    SplitData,
    missingness_report,
    outcome_survival,
    prepare_imputed,
    prepare_strict,
)
from hit_probability.models import TrainedModel, train_nn, train_rf_classifier, train_rf_regressor

logger = logging.getLogger(__name__)

Curves = dict[str, tuple[np.ndarray, np.ndarray]]


def _evaluate_models(
    split: SplitData, method: str, config: Config
) -> tuple[list[dict], dict[str, TrainedModel]]:
    """Train the RF classifier (and NN, if enabled) on one split and score both."""
    logger.info("[%s] train %s, test %s, %d features", method, f"{len(split.X_train):,}",
                f"{len(split.X_test):,}", split.X_train.shape[1])  # fmt: skip
    rows, trained = [], {}

    trained["Random forest"] = train_rf_classifier(split, config.n_estimators)
    if config.run_nn:
        trained["Neural network"] = train_nn(split, config.epochs, config.batch_size)

    for name, model in trained.items():
        rows.append(score(name, method, split.y_test, model.p_hit, model.train_time_s))
        logger.info("[%s] %s accuracy %.4f, ROC AUC %.4f", method, name,
                    rows[-1]["accuracy"], rows[-1]["roc_auc"])  # fmt: skip
    return rows, trained


def _curves(split: SplitData, trained: dict[str, TrainedModel], tag: str) -> Curves:
    return {f"{name} ({tag})": (split.y_test, m.p_hit) for name, m in trained.items()}


def run(config: Config) -> pd.DataFrame:
    """Run every experiment and write results to ``config.results_dir``."""
    config.results_dir.mkdir(parents=True, exist_ok=True)

    pitches = load_statcast(
        config.start_date, config.end_date, config.cache_path, config.refresh_data
    )
    validate_columns(pitches)
    coverage = coverage_report(pitches, config.end_date)

    if config.sample_frac:
        pitches = pitches.sample(frac=config.sample_frac, random_state=RANDOM_STATE)
        logger.info("Sampled %.0f%% of pitches for a quick run", 100 * config.sample_frac)

    bip = add_spray_angle(prepare_batted_balls(pitches))

    # Diagnostics: what the strict method throws away.
    missingness_report(bip, VIDEO_FEATURES).to_csv(config.results_dir / "missingness.csv")
    outcome_survival(bip, VIDEO_FEATURES).to_csv(config.results_dir / "outcome_survival.csv")

    rows: list[dict] = []
    curves: Curves = {}

    # Method A: strict dropna, video features.
    strict = prepare_strict(bip, VIDEO_FEATURES, CATEGORICAL_FEATURES, config.test_size)
    baseline = majority_baseline_accuracy(strict.y_test)
    logger.info("Always-predict-'out' accuracy: %.4f (every model has to beat this)", baseline)

    video_rf = train_rf_regressor(strict, config.n_estimators)
    logger.info("Video's RF regressor: R^2 %.4f (not accuracy), accuracy at 0.5 %.4f",
                video_rf["r2"], video_rf["accuracy_at_0.5"])  # fmt: skip

    strict_rows, strict_models = _evaluate_models(strict, "Strict (dropna)", config)
    rows += strict_rows
    curves.update(_curves(strict, strict_models, "video features"))

    importance = permutation_importance_table(
        strict_models["Random forest"].model, strict, config.permutation_sample
    )
    nn_history = strict_models["Neural network"].history if config.run_nn else None

    # Method B: median / mode imputation, video features.
    imputed = prepare_imputed(bip, VIDEO_FEATURES, CATEGORICAL_FEATURES, config.test_size)
    rows += _evaluate_models(imputed, "Median / mode", config)[0]

    # Leak-free feature set.
    if config.run_leak_free:
        leak_free = prepare_strict(bip, LEAK_FREE_FEATURES, CATEGORICAL_FEATURES, config.test_size)
        lf_rows, lf_models = _evaluate_models(leak_free, "Strict, leak-free", config)
        rows += lf_rows
        curves.update(_curves(leak_free, lf_models, "leak-free"))

    comparison = pd.DataFrame(rows)
    _write_results(comparison, baseline, video_rf, coverage, len(bip), config)

    if config.make_dashboard:
        from hit_probability.dashboard import build_dashboard

        build_dashboard(bip, comparison, baseline, curves, importance, nn_history, coverage,
                        config)  # fmt: skip
    return comparison


def _write_results(
    comparison: pd.DataFrame,
    baseline: float,
    video_rf: dict,
    coverage: dict,
    n_bip: int,
    config: Config,
) -> None:
    comparison.to_csv(config.results_dir / "model_comparison.csv", index=False)
    (config.results_dir / "model_comparison.md").write_text(
        to_markdown_table(comparison) + "\n", encoding="utf-8"
    )
    summary = {
        "date_range": [config.start_date, config.end_date],
        "coverage": coverage,
        "sample_frac": config.sample_frac,
        "batted_balls": n_bip,
        "majority_baseline_accuracy": baseline,
        "video_rf_regressor": video_rf,
    }
    summary_path = config.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Results written to %s", config.results_dir.resolve())
