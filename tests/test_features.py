from hit_probability.config import CATEGORICAL_FEATURES, VIDEO_FEATURES
from hit_probability.features import outcome_survival, prepare_imputed, prepare_strict


def test_strict_has_no_missing_values_and_is_numeric(batted_balls):
    split = prepare_strict(batted_balls, VIDEO_FEATURES, CATEGORICAL_FEATURES)
    for X in (split.X_train, split.X_test):
        assert not X.isna().any().any()
        assert all(dtype.kind == "f" for dtype in X.dtypes)
    assert split.n_rows < len(batted_balls)


def test_imputed_keeps_every_row_and_aligns_columns(batted_balls):
    split = prepare_imputed(batted_balls, VIDEO_FEATURES, CATEGORICAL_FEATURES)
    assert split.n_rows == len(batted_balls)
    assert list(split.X_train.columns) == list(split.X_test.columns)
    assert not split.X_train.isna().any().any()
    assert not split.X_test.isna().any().any()


def test_split_is_stratified(batted_balls):
    split = prepare_strict(batted_balls, VIDEO_FEATURES, CATEGORICAL_FEATURES)
    assert abs(split.y_train.mean() - split.y_test.mean()) < 0.02


def test_survival_flags_dropped_home_runs(batted_balls):
    table = outcome_survival(batted_balls, VIDEO_FEATURES)
    assert table.loc["home_run", "after_strict_dropna"] == 0
