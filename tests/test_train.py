from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.train as train_module


class FakeModel:
    feature_importances_ = np.array([0.7, 0.3])

    def predict_proba(self, dataframe):
        positive = np.where(dataframe["feature_a"].to_numpy() > 0.5, 0.8, 0.1)
        return np.column_stack([1 - positive, positive])


def training_dataset() -> pd.DataFrame:
    target = np.array([1 if index % 5 == 0 else 0 for index in range(100)])
    return pd.DataFrame(
        {
            "ACCOUNT_ID": range(100),
            "feature_a": target.astype(float),
            "feature_b": np.linspace(0, 1, 100),
            "TARGET": target,
        }
    )


def test_find_best_f1_threshold_finds_separating_threshold() -> None:
    threshold = train_module.find_best_f1_threshold(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    assert threshold == pytest.approx(0.8)


def test_train_writes_model_metadata_and_temporal_dataset(
    monkeypatch,
    tmp_path,
) -> None:
    dataset = training_dataset()
    leaderboard = pd.DataFrame(
        [
            {
                "model": "FakeModel",
                "roc_auc": 1.0,
                "f1": 1.0,
                "threshold": 0.5,
                "selection_score": 2.0,
            }
        ]
    )
    dumped = []

    monkeypatch.setattr(train_module, "load_transactions", lambda path: object())
    monkeypatch.setattr(train_module, "clean_data", lambda data: data)
    monkeypatch.setattr(
        train_module,
        "build_temporal_account_dataset",
        lambda **kwargs: dataset.copy(),
    )
    monkeypatch.setattr(
        train_module,
        "fit_best_model",
        lambda X, y: (FakeModel(), 0.5, leaderboard),
    )
    monkeypatch.setattr(
        train_module.joblib,
        "dump",
        lambda value, path: dumped.append((value, path)),
    )
    monkeypatch.setattr(
        train_module,
        "TEMPORAL_DATASET_FILE",
        tmp_path / "data" / "temporal.csv",
    )
    monkeypatch.setattr(
        train_module,
        "MODEL_FILE",
        tmp_path / "models" / "model.pkl",
    )
    monkeypatch.setattr(
        train_module,
        "METADATA_FILE",
        tmp_path / "models" / "metadata.pkl",
    )

    train_module.train(enable_mlflow=False)

    assert (tmp_path / "data" / "temporal.csv").exists()
    assert len(dumped) == 2
    metadata = dumped[1][0]
    assert metadata["algorithm"] == "FakeModel"
    assert metadata["features"] == ["feature_a", "feature_b"]
    assert metadata["decision_threshold"] == pytest.approx(0.5)
    assert metadata["roc_auc"] == pytest.approx(1.0)


def test_train_rejects_single_class_target(monkeypatch, tmp_path) -> None:
    dataset = training_dataset().assign(TARGET=0)
    monkeypatch.setattr(train_module, "load_transactions", lambda path: object())
    monkeypatch.setattr(train_module, "clean_data", lambda data: data)
    monkeypatch.setattr(
        train_module,
        "build_temporal_account_dataset",
        lambda **kwargs: dataset,
    )
    monkeypatch.setattr(
        train_module,
        "TEMPORAL_DATASET_FILE",
        tmp_path / "temporal.csv",
    )

    with pytest.raises(ValueError, match="une seule classe"):
        train_module.train(enable_mlflow=False)
