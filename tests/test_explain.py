import numpy as np
import pytest
import shap

import src.explain as explain_module


class FakeModel:
    def predict_proba(self, dataframe):
        return np.array([[0.2, 0.8]])


def test_load_artifacts_loads_model_and_metadata(monkeypatch) -> None:
    fake_model = object()
    fake_metadata = {"features": ["feature_a"]}

    def fake_load(path):
        if path == explain_module.MODEL_FILE:
            return fake_model
        if path == explain_module.METADATA_FILE:
            return fake_metadata
        raise AssertionError(f"Unexpected artifact path: {path}")

    monkeypatch.setattr(explain_module.joblib, "load", fake_load)

    model, metadata = explain_module.load_artifacts()

    assert model is fake_model
    assert metadata is fake_metadata


@pytest.fixture
def account_data() -> dict:
    return {"feature_a": 2.0, "feature_b": 4.0}


def test_explain_account_handles_multiclass_shap_values(
    monkeypatch,
    account_data: dict,
) -> None:
    metadata = {
        "features": ["feature_a", "feature_b"],
        "decision_threshold": 0.4,
    }
    monkeypatch.setattr(
        explain_module,
        "load_artifacts",
        lambda: (FakeModel(), metadata),
    )

    class FakeExplainer:
        def __call__(self, dataframe):
            return shap.Explanation(
                values=np.array([[[0.1, 0.3], [-0.2, -0.4]]]),
                base_values=np.array([[0.8, 0.2]]),
                data=dataframe.to_numpy(),
                feature_names=list(dataframe.columns),
            )

    monkeypatch.setattr(
        explain_module.shap,
        "TreeExplainer",
        lambda model: FakeExplainer(),
    )

    result = explain_module.explain_account(account_data)

    assert result["prediction"] == 1
    assert result["risk_score"] == pytest.approx(80.0)
    assert list(result["contributions"]["feature"]) == [
        "feature_b",
        "feature_a",
    ]
    assert result["contributions"]["shap_value"].tolist() == [
        -0.4,
        0.3,
    ]


def test_explain_account_reports_missing_features(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        explain_module,
        "load_artifacts",
        lambda: (FakeModel(), {"features": ["required_feature"]}),
    )

    with pytest.raises(ValueError, match="required_feature"):
        explain_module.explain_account({})
