from __future__ import annotations

import joblib
import pandas as pd

from src.config import (
    METADATA_FILE,
    MODEL_FILE,
)


def load_artifacts():
    model = joblib.load(MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)

    return model, metadata


def predict_account(data: dict) -> dict:
    model, metadata = load_artifacts()

    expected_features = metadata["features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Caractéristiques manquantes : "
            f"{missing_features}"
        )

    input_df = pd.DataFrame(
        [
            {
                feature: data[feature]
                for feature in expected_features
            }
        ]
    )

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    threshold = float(
        metadata.get(
            "decision_threshold",
            0.5,
        )
    )

    return {
        "prediction": int(
            probability >= threshold
        ),
        "risk_score": round(
            probability * 100,
            2,
        ),
        "decision_threshold": round(
            threshold * 100,
            2,
        ),
    }