from __future__ import annotations

import math

import joblib
import pandas as pd
import pytest

from src.config import (
    METADATA_FILE,
    MODEL_FILE,
    TEMPORAL_DATASET_FILE,
)
from src.predict import predict_account


@pytest.mark.smoke
def test_saved_model_predicts_real_account_row() -> None:
    missing_artifacts = [
        path
        for path in [MODEL_FILE, METADATA_FILE, TEMPORAL_DATASET_FILE]
        if not path.exists()
    ]
    if missing_artifacts:
        pytest.skip(
            "Generate local model artifacts with `python -m src.train`: "
            f"{missing_artifacts}"
        )

    model = joblib.load(MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)
    expected_features = metadata["features"]
    dataset = pd.read_csv(TEMPORAL_DATASET_FILE, nrows=1)

    assert not dataset.empty
    assert expected_features
    assert set(expected_features).issubset(dataset.columns)
    assert hasattr(model, "predict_proba")

    account_data = dataset.loc[0, expected_features].to_dict()
    result = predict_account(account_data)

    assert result["prediction"] in {0, 1}
    assert math.isfinite(result["risk_score"])
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["decision_threshold"] == pytest.approx(
        float(metadata["decision_threshold"]) * 100,
        abs=0.01,
    )
