"""Random forest and neural network training."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from hit_probability.config import RANDOM_STATE
from hit_probability.features import SplitData

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    model: Any
    p_hit: np.ndarray
    train_time_s: float
    history: dict[str, list[float]] = field(default_factory=dict)


def train_rf_regressor(split: SplitData, n_estimators: int = 100) -> dict[str, float]:
    """The video's original model, kept for reference.

    It predicts a value in [0, 1] and is scored with R^2 and RMSE, which are regression
    metrics, not classification accuracy.
    """
    start = time.time()
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(split.X_train, split.y_train)
    train_time = time.time() - start

    preds = rf.predict(split.X_test)
    return {
        "r2": r2_score(split.y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(split.y_test, preds))),
        "accuracy_at_0.5": float(np.mean((preds >= 0.5) == split.y_test)),
        "train_time_s": train_time,
    }


def train_rf_classifier(split: SplitData, n_estimators: int = 100) -> TrainedModel:
    start = time.time()
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(split.X_train, split.y_train)
    train_time = time.time() - start
    return TrainedModel(rf, rf.predict_proba(split.X_test)[:, 1], train_time)


def build_nn(n_features: int):
    """Input -> 64 ReLU units -> 2-unit softmax (P(out), P(hit)), as in the video."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_nn(
    split: SplitData, epochs: int = 20, batch_size: int = 100, verbose: int = 0
) -> TrainedModel:
    from tensorflow import keras

    keras.utils.set_random_seed(RANDOM_STATE)

    # Neural nets train poorly on unscaled inputs; the scaler is fit on train only.
    scaler = StandardScaler()
    X_train = np.asarray(scaler.fit_transform(split.X_train), dtype=np.float32)
    X_test = np.asarray(scaler.transform(split.X_test), dtype=np.float32)
    y_train = np.asarray(split.y_train, dtype=np.int32)

    model = build_nn(X_train.shape[1])
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=verbose,
    )
    train_time = time.time() - start

    p_hit = model.predict(X_test, verbose=0)[:, 1]
    return TrainedModel(model, p_hit, train_time, history=history.history)
