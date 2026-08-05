from __future__ import annotations

import joblib
import pandas as pd
import shap

from src.config import (
    METADATA_FILE,
    MODEL_FILE,
)


def load_artifacts():
    """Charge le modèle et les métadonnées."""

    model = joblib.load(MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)

    return model, metadata


def explain_account(data: dict) -> dict:
    """
    Produit une explication SHAP pour un seul compte.
    """

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

    prediction = int(
        probability >= threshold
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        input_df
    )

    if shap_values.values.ndim == 3:
        positive_values = shap_values.values[0, :, 1]
        positive_base_value = shap_values.base_values[0, 1]
        shap_explanation = shap.Explanation(
            values=positive_values,
            base_values=positive_base_value,
            data=input_df.iloc[0].values,
            feature_names=expected_features,
        )
    else:
        positive_values = shap_values.values[0]
        shap_explanation = shap_values[0]

    contributions = pd.DataFrame(
        {
            "feature": expected_features,
            "value": input_df.iloc[0].values,
            "shap_value": positive_values,
        }
    )

    contributions["absolute_contribution"] = (
        contributions["shap_value"].abs()
    )

    contributions = contributions.sort_values(
        "absolute_contribution",
        ascending=False,
    )

    return {
        "prediction": prediction,
        "risk_score": round(
            probability * 100,
            2,
        ),
        "decision_threshold": round(
            threshold * 100,
            2,
        ),
        "contributions": contributions,
        "shap_explanation": shap_explanation,
    }
