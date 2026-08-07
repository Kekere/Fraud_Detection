import numpy as np
import pytest

import src.predict as predict_module


class FakeFraudModel:
    """Modèle simulé pour tester la logique de prédiction."""

    def predict(self, dataframe):
        assert len(dataframe) == 1
        return np.array([1])

    def predict_proba(self, dataframe):
        assert len(dataframe) == 1

        # Probabilité :
        # classe 0 = 0,13
        # classe 1 = 0,87
        return np.array([[0.13, 0.87]])


class FakeNormalModel:
    """Modèle simulant une prédiction non frauduleuse."""

    def predict(self, dataframe):
        return np.array([0])

    def predict_proba(self, dataframe):
        return np.array([[0.92, 0.08]])


def test_load_artifacts_loads_model_and_metadata(monkeypatch) -> None:
    fake_model = object()
    fake_metadata = {"features": ["feature_a"]}

    def fake_load(path):
        if path == predict_module.MODEL_FILE:
            return fake_model
        if path == predict_module.METADATA_FILE:
            return fake_metadata
        raise AssertionError(f"Unexpected artifact path: {path}")

    monkeypatch.setattr(predict_module.joblib, "load", fake_load)

    model, metadata = predict_module.load_artifacts()

    assert model is fake_model
    assert metadata is fake_metadata


@pytest.fixture
def account_data() -> dict:
    return {
        "n_transactions": 25,
        "total_amount": 75000.0,
        "avg_amount": 3000.0,
        "median_amount": 1500.0,
        "std_amount": 4200.0,
        "min_amount": 25.0,
        "max_amount": 25000.0,
        "n_counterparties": 7,
        "avg_time_between_tx": 1800.0,
    }


@pytest.fixture
def expected_features() -> list[str]:
    return [
        "n_transactions",
        "total_amount",
        "avg_amount",
        "median_amount",
        "std_amount",
        "min_amount",
        "max_amount",
        "n_counterparties",
        "avg_time_between_tx",
    ]


def test_predict_account_returns_expected_structure(
    monkeypatch,
    account_data: dict,
    expected_features: list[str],
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_artifacts",
        lambda: (
            FakeFraudModel(),
            {"features": expected_features, "decision_threshold": 0.4},
        ),
    )

    result = predict_module.predict_account(account_data)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "risk_score" in result
    assert "decision_threshold" in result
    assert result["decision_threshold"] == pytest.approx(40.0)


def test_predict_account_returns_fraud_prediction(
    monkeypatch,
    account_data: dict,
    expected_features: list[str],
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_artifacts",
        lambda: (FakeFraudModel(), {"features": expected_features}),
    )

    result = predict_module.predict_account(account_data)

    assert result["prediction"] == 1
    assert result["risk_score"] == pytest.approx(87.0)


def test_predict_account_returns_normal_prediction(
    monkeypatch,
    account_data: dict,
    expected_features: list[str],
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_artifacts",
        lambda: (FakeNormalModel(), {"features": expected_features}),
    )

    result = predict_module.predict_account(account_data)

    assert result["prediction"] == 0
    assert result["risk_score"] == pytest.approx(8.0)


def test_predict_account_uses_custom_decision_threshold(
    monkeypatch,
    account_data: dict,
    expected_features: list[str],
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_artifacts",
        lambda: (
            FakeNormalModel(),
            {"features": expected_features, "decision_threshold": 0.05},
        ),
    )

    result = predict_module.predict_account(account_data)

    assert result["prediction"] == 1
    assert result["risk_score"] == pytest.approx(8.0)
    assert result["decision_threshold"] == pytest.approx(5.0)


def test_predict_account_uses_metadata_feature_order(
    monkeypatch,
    account_data: dict,
    expected_features: list[str],
) -> None:
    class FeatureOrderModel:
        def predict(self, dataframe):
            assert list(dataframe.columns) == expected_features
            return np.array([0])

        def predict_proba(self, dataframe):
            assert list(dataframe.columns) == expected_features
            return np.array([[0.75, 0.25]])

    # Données volontairement fournies dans un ordre différent.
    reversed_data = dict(reversed(list(account_data.items())))

    monkeypatch.setattr(
        predict_module,
        "load_artifacts",
        lambda: (FeatureOrderModel(), {"features": expected_features}),
    )

    result = predict_module.predict_account(reversed_data)

    assert result["prediction"] == 0
    assert result["risk_score"] == pytest.approx(25.0)


def test_predict_account_ignores_extra_fields(
    monkeypatch,
    account_data: dict,
    expected_features: list[str],
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_artifacts",
        lambda: (FakeFraudModel(), {"features": expected_features}),
    )

    data_with_extra_field = {
        **account_data,
        "ACCOUNT_ID": "A001",
        "irrelevant_field": "ignored",
    }

    result = predict_module.predict_account(data_with_extra_field)

    assert result["prediction"] == 1
    assert result["risk_score"] == pytest.approx(87.0)


def test_predict_account_raises_error_when_feature_is_missing(
    monkeypatch,
    account_data: dict,
    expected_features: list[str],
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_artifacts",
        lambda: (FakeFraudModel(), {"features": expected_features}),
    )

    incomplete_data = account_data.copy()
    incomplete_data.pop("max_amount")

    with pytest.raises(ValueError, match="max_amount"):
        predict_module.predict_account(incomplete_data)
