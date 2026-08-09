from fastapi.testclient import TestClient

import app.api as api_module

client = TestClient(api_module.app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_root_redirects_to_documentation() -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_favicon_does_not_return_404() -> None:
    response = client.get("/favicon.ico")

    assert response.status_code == 204


def test_predict_returns_result_and_request_id(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "predict_account",
        lambda features: {
            "prediction": 1,
            "risk_score": 87.0,
            "decision_threshold": 40.0,
        },
    )

    response = client.post(
        "/predict",
        headers={"X-Request-ID": "test-request"},
        json={"account_id": "A001", "features": {"amount": 125.5}},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"
    assert response.json() == {
        "request_id": "test-request",
        "prediction": 1,
        "risk_score": 87.0,
        "decision_threshold": 40.0,
    }


def test_predict_loads_features_from_account_id(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "features_for_account",
        lambda account_id: {"amount": 125.5},
    )
    monkeypatch.setattr(
        api_module,
        "predict_account",
        lambda features: {
            "prediction": 0,
            "risk_score": 12.0,
            "decision_threshold": 40.0,
        },
    )

    response = client.post("/predict", json={"account_id": "1"})

    assert response.status_code == 200
    assert response.json()["prediction"] == 0


def test_predict_rejects_unknown_account(monkeypatch) -> None:
    def reject_account(account_id):
        raise ValueError(f"Compte introuvable : {account_id}")

    monkeypatch.setattr(api_module, "features_for_account", reject_account)

    response = client.post("/predict", json={"account_id": "unknown"})

    assert response.status_code == 422
    assert "Compte introuvable" in response.json()["detail"]


def test_predict_returns_422_for_missing_features(monkeypatch) -> None:
    def reject_prediction(features):
        raise ValueError("Caractéristiques manquantes : ['amount']")

    monkeypatch.setattr(api_module, "predict_account", reject_prediction)

    response = client.post("/predict", json={"features": {"count": 2}})

    assert response.status_code == 422
    assert "amount" in response.json()["detail"]
    assert response.json()["request_id"]


def test_predict_validates_payload() -> None:
    response = client.post("/predict", json={"features": {"amount": "invalid"}})

    assert response.status_code == 422


def test_predict_requires_account_or_features() -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422
