from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "TX_ID",
    "SENDER_ACCOUNT_ID",
    "RECEIVER_ACCOUNT_ID",
    "TX_TYPE",
    "TX_AMOUNT",
    "TIMESTAMP",
    "IS_FRAUD",
]


def load_transactions(filepath: str | Path) -> pd.DataFrame:
    """Charge le dataset et normalise les types."""

    df = pd.read_csv(filepath)

    df.columns = [
        column.strip().upper()
        for column in df.columns
    ]

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes requises absentes : {missing_columns}. "
            f"Colonnes disponibles : {df.columns.tolist()}"
        )

    df["TIMESTAMP"] = pd.to_numeric(
        df["TIMESTAMP"],
        errors="coerce",
    )

    df["TX_AMOUNT"] = pd.to_numeric(
        df["TX_AMOUNT"],
        errors="coerce",
    )

    if pd.api.types.is_bool_dtype(df["IS_FRAUD"]):
        df["IS_FRAUD"] = df["IS_FRAUD"].astype(int)

    else:
        df["IS_FRAUD"] = (
            df["IS_FRAUD"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                }
            )
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et valide les transactions."""

    df = df.copy()

    df = df.drop_duplicates(
        subset=["TX_ID"]
    )

    df = df.dropna(
        subset=[
            "TX_ID",
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
            "TX_TYPE",
            "TX_AMOUNT",
            "TIMESTAMP",
            "IS_FRAUD",
        ]
    )

    df = df[df["TX_AMOUNT"] >= 0]

    df["TIMESTAMP"] = df["TIMESTAMP"].astype(int)
    df["IS_FRAUD"] = df["IS_FRAUD"].astype(int)

    if not set(df["IS_FRAUD"].unique()).issubset({0, 1}):
        raise ValueError(
            "IS_FRAUD doit contenir uniquement 0 et 1."
        )

    return (
        df.sort_values(
            [
                "TIMESTAMP",
                "TX_ID",
            ]
        )
        .reset_index(drop=True)
    )