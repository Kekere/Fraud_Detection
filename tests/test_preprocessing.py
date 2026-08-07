from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import clean_data, load_transactions


def valid_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TX_ID": [2, 1],
            "SENDER_ACCOUNT_ID": [10, 11],
            "RECEIVER_ACCOUNT_ID": [20, 21],
            "TX_TYPE": ["TRANSFER", "PAYMENT"],
            "TX_AMOUNT": [25.5, 10.0],
            "TIMESTAMP": [2, 1],
            "IS_FRAUD": [False, True],
        }
    )


def test_load_transactions_normalizes_columns_and_types(
    tmp_path,
) -> None:
    source = valid_transactions().rename(
        columns={"TX_ID": " tx_id "}
    )
    path = tmp_path / "transactions.csv"
    source.to_csv(path, index=False)

    result = load_transactions(path)

    assert "TX_ID" in result.columns
    assert result["IS_FRAUD"].tolist() == [0, 1]
    assert pd.api.types.is_numeric_dtype(result["TIMESTAMP"])
    assert pd.api.types.is_numeric_dtype(result["TX_AMOUNT"])


def test_load_transactions_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"TX_ID": [1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Colonnes requises absentes"):
        load_transactions(path)


def test_clean_data_deduplicates_filters_and_sorts() -> None:
    data = valid_transactions()
    duplicate = data.iloc[[0]].copy()
    negative = data.iloc[[0]].assign(TX_ID=3, TX_AMOUNT=-1)
    missing = data.iloc[[0]].assign(TX_ID=4, TX_TYPE=None)

    result = clean_data(
        pd.concat([data, duplicate, negative, missing], ignore_index=True)
    )

    assert result["TX_ID"].tolist() == [1, 2]
    assert result["TIMESTAMP"].tolist() == [1, 2]
    assert result["IS_FRAUD"].tolist() == [1, 0]


def test_clean_data_rejects_invalid_fraud_labels() -> None:
    data = valid_transactions()
    data["IS_FRAUD"] = [0, 2]

    with pytest.raises(ValueError, match="uniquement 0 et 1"):
        clean_data(data)
