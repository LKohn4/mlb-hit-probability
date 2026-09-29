import numpy as np
import pytest

from hit_probability.config import CATEGORICAL_FEATURES, VIDEO_FEATURES
from hit_probability.evaluate import majority_baseline_accuracy, score, to_markdown_table
from hit_probability.features import prepare_strict
from hit_probability.models import train_rf_classifier, train_rf_regressor


@pytest.fixture
def split(batted_balls):
    return prepare_strict(batted_balls, VIDEO_FEATURES, CATEGORICAL_FEATURES)


def test_rf_classifier_returns_probabilities(split):
    trained = train_rf_classifier(split, n_estimators=10)
    assert trained.p_hit.shape == split.y_test.shape
    assert ((trained.p_hit >= 0) & (trained.p_hit <= 1)).all()


def test_rf_regressor_reports_video_metrics(split):
    metrics = train_rf_regressor(split, n_estimators=10)
    assert {"r2", "rmse", "accuracy_at_0.5", "train_time_s"} <= metrics.keys()


def test_nn_trains(split):
    pytest.importorskip("tensorflow")
    from hit_probability.models import train_nn

    trained = train_nn(split, epochs=1, batch_size=256)
    assert trained.p_hit.shape == split.y_test.shape
    assert "val_accuracy" in trained.history


def test_score_handles_extreme_probabilities():
    y = np.array([0, 1, 0, 1])
    result = score("m", "x", y, np.array([0.0, 1.0, 0.0, 1.0]), 1.0)
    assert result["accuracy"] == 1.0
    assert np.isfinite(result["log_loss"])


def test_majority_baseline():
    assert majority_baseline_accuracy(np.array([0, 0, 0, 1])) == 0.75


def test_markdown_table_shape():
    import pandas as pd

    md = to_markdown_table(pd.DataFrame({"a": [1.0], "b": ["x"]}))
    assert md.splitlines()[0] == "| a | b |"
    assert len(md.splitlines()) == 3
