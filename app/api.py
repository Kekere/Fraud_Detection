from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import TEMPORAL_DATASET_FILE
from src.predict import predict_account


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "event_data", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("fraud_api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("api_started", extra={"event_data": {"version": "1.0.0"}})
    yield
    logger.info("api_stopped")


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="API de prédiction du risque de fraude.",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"account_id": "1"},
                {
                    "features": {
                        "n_transactions": 12,
                        "total_outgoing_amount": 2109.6,
                        "avg_outgoing_amount": 175.8,
                    }
                },
            ]
        },
    )

    account_id: str | None = Field(
        default=None,
        description="Identifiant présent dans GET /accounts.",
    )
    features: dict[str, Annotated[float, Field(allow_inf_nan=False)]] | None = Field(
        default=None,
        description=(
            "Caractéristiques calculées du modèle. À omettre lorsqu'account_id est fourni."
        ),
    )

    @model_validator(mode="after")
    def require_account_or_features(self) -> PredictionRequest:
        if self.account_id is None and self.features is None:
            raise ValueError("Fournissez account_id ou features")
        return self


class PredictionResponse(BaseModel):
    request_id: str
    prediction: int
    risk_score: float
    decision_threshold: float


@lru_cache(maxsize=1)
def load_account_features() -> pd.DataFrame:
    started_at = time.perf_counter()
    dataset = pd.read_csv(TEMPORAL_DATASET_FILE)
    logger.info(
        "dataset_loaded",
        extra={
            "event_data": {
                "rows": len(dataset),
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }
        },
    )
    return dataset


def features_for_account(account_id: str) -> dict[str, float]:
    dataset = load_account_features()
    account_rows = dataset.loc[dataset["ACCOUNT_ID"].astype(str) == account_id]
    if account_rows.empty:
        raise ValueError(f"Compte introuvable : {account_id}")
    return account_rows.iloc[0].drop(labels=["ACCOUNT_ID", "TARGET"], errors="ignore").to_dict()


@app.middleware("http")
async def log_request(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "request_failed",
            extra={
                "event_data": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                }
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Erreur interne de prédiction", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "event_data": {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        },
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/accounts")
def accounts() -> dict[str, list[str]]:
    dataset = load_account_features()
    return {"account_ids": dataset["ACCOUNT_ID"].astype(str).tolist()}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "account": {
                            "summary": "Prédire un compte du dataset",
                            "value": {"account_id": "1"},
                        },
                        "features": {
                            "summary": "Fournir des caractéristiques calculées",
                            "value": {
                                "features": {
                                    "n_transactions": 12,
                                    "total_outgoing_amount": 2109.6,
                                    "avg_outgoing_amount": 175.8,
                                }
                            },
                        },
                    }
                }
            }
        }
    },
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    logger.info(
        "prediction_started",
        extra={
            "event_data": {
                "request_id": request.state.request_id,
                "account_id": payload.account_id,
                "input_mode": "features" if payload.features is not None else "account_id",
            }
        },
    )
    try:
        features = (
            payload.features
            if payload.features is not None
            else features_for_account(payload.account_id or "")
        )
        result = predict_account(features)
    except ValueError as error:
        logger.warning(
            "prediction_rejected",
            extra={
                "event_data": {
                    "request_id": request.state.request_id,
                    "account_id": payload.account_id,
                    "reason": str(error),
                }
            },
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(error),
                "request_id": request.state.request_id,
            },
        )

    logger.info(
        "prediction_completed",
        extra={
            "event_data": {
                "request_id": request.state.request_id,
                "account_id": payload.account_id,
                "prediction": result["prediction"],
                "risk_score": result["risk_score"],
            }
        },
    )
    return PredictionResponse(request_id=request.state.request_id, **result)
