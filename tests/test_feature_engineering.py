import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    build_account_features,
    build_future_target,
    build_temporal_account_dataset,
    split_temporal_windows,
)


@pytest.fixture
def transactions_df() -> pd.DataFrame:
    """Crée un petit jeu de transactions contrôlé pour les tests."""

    return pd.DataFrame(
        {
            "ACCOUNT_ID": ["A001", "A001", "A001", "A002", "A002"],
            "TIMESTAMP": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 13:00:00",
                    "2026-01-02 08:00:00",
                    "2026-01-02 08:30:00",
                ]
            ),
            "AMOUNT": [100.0, 200.0, 300.0, 50.0, 150.0],
            "COUNTERPARTY_ID": ["C001", "C002", "C001", "C003", "C004"],
            "IS_FRAUD": [0, 0, 1, 0, 0],
        }
    )


def test_build_account_features_returns_one_row_per_account(
    transactions_df: pd.DataFrame,
) -> None:
    result = build_account_features(transactions_df.copy())

    assert len(result) == 2
    assert result["ACCOUNT_ID"].nunique() == 2
    assert set(result["ACCOUNT_ID"]) == {"A001", "A002"}


def test_build_account_features_contains_expected_columns(
    transactions_df: pd.DataFrame,
) -> None:
    result = build_account_features(transactions_df.copy())

    expected_columns = {
        "ACCOUNT_ID",
        "n_transactions",
        "total_amount",
        "avg_amount",
        "median_amount",
        "std_amount",
        "min_amount",
        "max_amount",
        "n_counterparties",
        "avg_time_between_tx",
        "TARGET",
    }

    assert expected_columns.issubset(result.columns)


def test_amount_aggregations_are_correct(
    transactions_df: pd.DataFrame,
) -> None:
    result = build_account_features(transactions_df.copy())

    account = result.loc[result["ACCOUNT_ID"] == "A001"].iloc[0]

    assert account["n_transactions"] == 3
    assert account["total_amount"] == pytest.approx(600.0)
    assert account["avg_amount"] == pytest.approx(200.0)
    assert account["median_amount"] == pytest.approx(200.0)
    assert account["min_amount"] == pytest.approx(100.0)
    assert account["max_amount"] == pytest.approx(300.0)


def test_number_of_counterparties_is_correct(
    transactions_df: pd.DataFrame,
) -> None:
    result = build_account_features(transactions_df.copy())

    account_a001 = result.loc[result["ACCOUNT_ID"] == "A001"].iloc[0]
    account_a002 = result.loc[result["ACCOUNT_ID"] == "A002"].iloc[0]

    # C001 apparaît deux fois, mais ne doit être compté qu'une fois.
    assert account_a001["n_counterparties"] == 2
    assert account_a002["n_counterparties"] == 2


def test_target_is_one_when_account_has_fraud(
    transactions_df: pd.DataFrame,
) -> None:
    result = build_account_features(transactions_df.copy())

    target_a001 = result.loc[result["ACCOUNT_ID"] == "A001", "TARGET"].iloc[0]
    target_a002 = result.loc[result["ACCOUNT_ID"] == "A002", "TARGET"].iloc[0]

    assert target_a001 == 1
    assert target_a002 == 0


def test_average_time_between_transactions_is_correct(
    transactions_df: pd.DataFrame,
) -> None:
    result = build_account_features(transactions_df.copy())

    account_a001 = result.loc[result["ACCOUNT_ID"] == "A001"].iloc[0]
    account_a002 = result.loc[result["ACCOUNT_ID"] == "A002"].iloc[0]

    # A001 : moyenne de 3 600 et 7 200 secondes = 5 400 secondes.
    assert account_a001["avg_time_between_tx"] == pytest.approx(5400.0)

    # A002 : 30 minutes = 1 800 secondes.
    assert account_a002["avg_time_between_tx"] == pytest.approx(1800.0)


def test_result_does_not_contain_missing_values(
    transactions_df: pd.DataFrame,
) -> None:
    result = build_account_features(transactions_df.copy())

    assert not result.isna().any().any()


def test_single_transaction_account_has_zero_standard_deviation() -> None:
    df = pd.DataFrame(
        {
            "ACCOUNT_ID": ["A003"],
            "TIMESTAMP": pd.to_datetime(["2026-01-03 10:00:00"]),
            "AMOUNT": [500.0],
            "COUNTERPARTY_ID": ["C005"],
            "IS_FRAUD": [0],
        }
    )

    result = build_account_features(df)
    account = result.iloc[0]

    # Pandas retourne NaN pour l'écart-type d'une seule observation.
    # Le fillna(0) du script doit le transformer en 0.
    assert account["std_amount"] == pytest.approx(0.0)
    assert account["avg_time_between_tx"] == pytest.approx(0.0)
    assert not np.isnan(account["std_amount"])


@pytest.fixture
def temporal_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TX_ID": range(1, 9),
            "SENDER_ACCOUNT_ID": [1, 1, 2, 2, 1, 2, 1, 2],
            "RECEIVER_ACCOUNT_ID": [2, 3, 1, 3, 2, 1, 3, 3],
            "TX_TYPE": ["TRANSFER"] * 8,
            "TX_AMOUNT": [10, 20, 15, 25, 30, 35, 40, 45],
            "TIMESTAMP": [0, 1, 0, 1, 2, 2, 3, 3],
            "IS_FRAUD": [0, 0, 0, 0, 0, 0, 1, 0],
        }
    )


def test_temporal_windows_do_not_overlap(
    temporal_transactions_df: pd.DataFrame,
) -> None:
    observation, prediction = split_temporal_windows(
        temporal_transactions_df,
        observation_start=0,
        observation_end=1,
        prediction_start=2,
        prediction_end=3,
    )

    assert observation["TIMESTAMP"].max() == 1
    assert prediction["TIMESTAMP"].min() == 2
    assert set(observation["TX_ID"]).isdisjoint(prediction["TX_ID"])


def test_future_target_uses_only_future_fraud() -> None:
    prediction = pd.DataFrame(
        {
            "TX_ID": [1, 2, 3],
            "SENDER_ACCOUNT_ID": [1, 1, 2],
            "IS_FRAUD": [0, 1, 0],
        }
    )

    target = build_future_target(prediction).set_index("ACCOUNT_ID")

    assert target.loc[1, "TARGET"] == 1
    assert target.loc[2, "TARGET"] == 0


def test_temporal_dataset_is_finite_and_has_future_target(
    temporal_transactions_df: pd.DataFrame,
) -> None:
    result = build_temporal_account_dataset(
        temporal_transactions_df,
        observation_start=0,
        observation_end=1,
        prediction_start=2,
        prediction_end=3,
        recent_windows=[1],
        min_history_transactions=2,
    )

    assert set(result["ACCOUNT_ID"]) == {1, 2}
    assert result.set_index("ACCOUNT_ID").loc[1, "TARGET"] == 1
    numeric = result.drop(columns=["ACCOUNT_ID"])
    assert np.isfinite(numeric.to_numpy()).all()
