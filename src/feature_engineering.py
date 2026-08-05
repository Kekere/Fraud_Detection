from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


def sanitize_feature_name(value: object) -> str:
    """Transforme une valeur en nom de colonne sûr."""

    name = str(value).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")

    return name or "unknown"


def entropy(values: pd.Series) -> float:
    """
    Calcule l'entropie de Shannon d'une série catégorielle.

    Une entropie élevée indique une distribution plus diversifiée.
    """

    probabilities = (
        values.value_counts(normalize=True)
        .to_numpy()
    )

    if len(probabilities) == 0:
        return 0.0

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


def amount_entropy(values: pd.Series) -> float:
    """
    Calcule une entropie approximative des montants.

    Les montants sont regroupés en classes quantiles.
    """

    if len(values) < 2 or values.nunique() < 2:
        return 0.0

    number_of_bins = min(
        10,
        int(math.sqrt(len(values))),
    )

    try:
        bins = pd.qcut(
            values,
            q=number_of_bins,
            duplicates="drop",
        )

    except ValueError:
        return 0.0

    return entropy(bins)


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Division protégée contre les divisions par zéro."""

    denominator = denominator.replace(0, np.nan)

    ratio = numerator / denominator

    return (
        ratio
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def split_temporal_windows(
    df: pd.DataFrame,
    observation_start: int,
    observation_end: int,
    prediction_start: int,
    prediction_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les fenêtres historique et future."""

    if observation_end >= prediction_start:
        raise ValueError(
            "La fenêtre d'observation doit se terminer "
            "avant le début de la fenêtre de prédiction."
        )

    observation_df = df.loc[
        df["TIMESTAMP"].between(
            observation_start,
            observation_end,
            inclusive="both",
        )
    ].copy()

    prediction_df = df.loc[
        df["TIMESTAMP"].between(
            prediction_start,
            prediction_end,
            inclusive="both",
        )
    ].copy()

    if observation_df.empty:
        raise ValueError(
            "La fenêtre d'observation est vide."
        )

    if prediction_df.empty:
        raise ValueError(
            "La fenêtre de prédiction est vide."
        )

    print(
        "Fenêtre d'observation :",
        observation_df["TIMESTAMP"].min(),
        "à",
        observation_df["TIMESTAMP"].max(),
    )

    print(
        "Fenêtre de prédiction :",
        prediction_df["TIMESTAMP"].min(),
        "à",
        prediction_df["TIMESTAMP"].max(),
    )

    print(
        "Transactions d'observation :",
        len(observation_df),
    )

    print(
        "Transactions de prédiction :",
        len(prediction_df),
    )

    return observation_df, prediction_df


def build_outgoing_features(
    observation_df: pd.DataFrame,
    observation_end: int,
    recent_windows: list[int],
    high_amount_quantile: float,
) -> pd.DataFrame:
    """Construit les caractéristiques des flux sortants."""

    df = observation_df.copy()

    df = df.sort_values(
        [
            "SENDER_ACCOUNT_ID",
            "TIMESTAMP",
            "TX_ID",
        ]
    )

    df["delta_time_steps"] = (
        df.groupby("SENDER_ACCOUNT_ID")[
            "TIMESTAMP"
        ]
        .diff()
    )

    high_amount_threshold = float(
        df["TX_AMOUNT"].quantile(
            high_amount_quantile
        )
    )

    df["is_high_amount"] = (
        df["TX_AMOUNT"]
        >= high_amount_threshold
    ).astype(int)

    features = (
        df.groupby("SENDER_ACCOUNT_ID")
        .agg(
            n_transactions=(
                "TX_ID",
                "count",
            ),
            total_outgoing_amount=(
                "TX_AMOUNT",
                "sum",
            ),
            avg_outgoing_amount=(
                "TX_AMOUNT",
                "mean",
            ),
            median_outgoing_amount=(
                "TX_AMOUNT",
                "median",
            ),
            std_outgoing_amount=(
                "TX_AMOUNT",
                "std",
            ),
            min_outgoing_amount=(
                "TX_AMOUNT",
                "min",
            ),
            max_outgoing_amount=(
                "TX_AMOUNT",
                "max",
            ),
            n_counterparties=(
                "RECEIVER_ACCOUNT_ID",
                "nunique",
            ),
            n_transaction_types=(
                "TX_TYPE",
                "nunique",
            ),
            n_active_steps=(
                "TIMESTAMP",
                "nunique",
            ),
            first_observed_step=(
                "TIMESTAMP",
                "min",
            ),
            last_observed_step=(
                "TIMESTAMP",
                "max",
            ),
            avg_steps_between_tx=(
                "delta_time_steps",
                "mean",
            ),
            min_steps_between_tx=(
                "delta_time_steps",
                "min",
            ),
            max_steps_between_tx=(
                "delta_time_steps",
                "max",
            ),
            n_high_amount_transactions=(
                "is_high_amount",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    features["activity_duration_steps"] = (
        features["last_observed_step"]
        - features["first_observed_step"]
    )

    features["transactions_per_active_step"] = (
        features["n_transactions"]
        / features["n_active_steps"].clip(lower=1)
    )

    features["counterparties_per_transaction"] = (
        features["n_counterparties"]
        / features["n_transactions"].clip(lower=1)
    )

    features["high_amount_transaction_ratio"] = (
        features["n_high_amount_transactions"]
        / features["n_transactions"].clip(lower=1)
    )

    # Coefficient de variation des montants.
    features["amount_coefficient_of_variation"] = (
        features["std_outgoing_amount"]
        / features["avg_outgoing_amount"].replace(
            0,
            np.nan,
        )
    )

    # Concentration envers la principale contrepartie.
    counterparty_counts = (
        df.groupby(
            [
                "SENDER_ACCOUNT_ID",
                "RECEIVER_ACCOUNT_ID",
            ]
        )
        .size()
        .reset_index(name="counterparty_tx_count")
    )

    concentration = (
        counterparty_counts
        .groupby("SENDER_ACCOUNT_ID")
        .agg(
            max_transactions_same_counterparty=(
                "counterparty_tx_count",
                "max",
            )
        )
        .reset_index()
        .rename(
            columns={
                "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    features = features.merge(
        concentration,
        on="ACCOUNT_ID",
        how="left",
    )

    features["top_counterparty_ratio"] = (
        features[
            "max_transactions_same_counterparty"
        ]
        / features["n_transactions"].clip(lower=1)
    )

    # Entropie des contreparties.
    counterparty_entropy = (
        df.groupby("SENDER_ACCOUNT_ID")[
            "RECEIVER_ACCOUNT_ID"
        ]
        .apply(entropy)
        .reset_index(
            name="counterparty_entropy"
        )
        .rename(
            columns={
                "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    features = features.merge(
        counterparty_entropy,
        on="ACCOUNT_ID",
        how="left",
    )

    # Entropie approximative des montants.
    amount_entropy_df = (
        df.groupby("SENDER_ACCOUNT_ID")[
            "TX_AMOUNT"
        ]
        .apply(amount_entropy)
        .reset_index(
            name="outgoing_amount_entropy"
        )
        .rename(
            columns={
                "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    features = features.merge(
        amount_entropy_df,
        on="ACCOUNT_ID",
        how="left",
    )

    # Nombre de transactions par type.
    type_counts = pd.crosstab(
        df["SENDER_ACCOUNT_ID"],
        df["TX_TYPE"],
    )

    type_counts.columns = [
        f"tx_type_count_{sanitize_feature_name(column)}"
        for column in type_counts.columns
    ]

    type_counts = (
        type_counts
        .reset_index()
        .rename(
            columns={
                "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    features = features.merge(
        type_counts,
        on="ACCOUNT_ID",
        how="left",
    )

    # Caractéristiques des fenêtres récentes.
    for window_size in recent_windows:
        window_start = max(
            int(df["TIMESTAMP"].min()),
            observation_end - window_size + 1,
        )

        recent_df = df.loc[
            df["TIMESTAMP"].between(
                window_start,
                observation_end,
                inclusive="both",
            )
        ].copy()

        recent_features = (
            recent_df
            .groupby("SENDER_ACCOUNT_ID")
            .agg(
                **{
                    f"n_transactions_last_{window_size}_steps": (
                        "TX_ID",
                        "count",
                    ),
                    f"amount_last_{window_size}_steps": (
                        "TX_AMOUNT",
                        "sum",
                    ),
                    f"avg_amount_last_{window_size}_steps": (
                        "TX_AMOUNT",
                        "mean",
                    ),
                    f"max_amount_last_{window_size}_steps": (
                        "TX_AMOUNT",
                        "max",
                    ),
                    f"counterparties_last_{window_size}_steps": (
                        "RECEIVER_ACCOUNT_ID",
                        "nunique",
                    ),
                }
            )
            .reset_index()
            .rename(
                columns={
                    "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
                }
            )
        )

        features = features.merge(
            recent_features,
            on="ACCOUNT_ID",
            how="left",
        )

        # Comparaison comportement récent / historique.
        features[
            f"recent_tx_frequency_ratio_{window_size}"
        ] = safe_ratio(
            features[
                f"n_transactions_last_{window_size}_steps"
            ],
            features["n_transactions"],
        )

        features[
            f"recent_amount_ratio_{window_size}"
        ] = safe_ratio(
            features[
                f"amount_last_{window_size}_steps"
            ],
            features["total_outgoing_amount"],
        )

        features[
            f"recent_avg_amount_change_{window_size}"
        ] = safe_ratio(
            features[
                f"avg_amount_last_{window_size}_steps"
            ],
            features["avg_outgoing_amount"],
        )

    features = features.drop(
        columns=[
            "first_observed_step",
            "last_observed_step",
        ]
    )

    return features


def build_incoming_features(
    observation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construit les caractéristiques des flux entrants."""

    incoming = (
        observation_df
        .groupby("RECEIVER_ACCOUNT_ID")
        .agg(
            n_incoming_transactions=(
                "TX_ID",
                "count",
            ),
            total_incoming_amount=(
                "TX_AMOUNT",
                "sum",
            ),
            avg_incoming_amount=(
                "TX_AMOUNT",
                "mean",
            ),
            max_incoming_amount=(
                "TX_AMOUNT",
                "max",
            ),
            n_unique_senders=(
                "SENDER_ACCOUNT_ID",
                "nunique",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "RECEIVER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    return incoming


def build_network_features(
    observation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit des caractéristiques simples du réseau.

    Elles ne nécessitent pas de bibliothèque de graphes.
    """

    outgoing_degree = (
        observation_df
        .groupby("SENDER_ACCOUNT_ID")[
            "RECEIVER_ACCOUNT_ID"
        ]
        .nunique()
        .reset_index(
            name="out_degree"
        )
        .rename(
            columns={
                "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    incoming_degree = (
        observation_df
        .groupby("RECEIVER_ACCOUNT_ID")[
            "SENDER_ACCOUNT_ID"
        ]
        .nunique()
        .reset_index(
            name="in_degree"
        )
        .rename(
            columns={
                "RECEIVER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    features = outgoing_degree.merge(
        incoming_degree,
        on="ACCOUNT_ID",
        how="outer",
    )

    features["total_degree"] = (
        features["out_degree"].fillna(0)
        + features["in_degree"].fillna(0)
    )

    features["out_in_degree_ratio"] = safe_ratio(
        features["out_degree"].fillna(0),
        features["in_degree"].fillna(0),
    )

    return features


def build_new_counterparty_features(
    observation_df: pd.DataFrame,
    observation_end: int,
    recent_window: int = 20,
) -> pd.DataFrame:
    """
    Mesure combien de contreparties observées récemment
    n'étaient pas présentes dans l'historique ancien.
    """

    recent_start = observation_end - recent_window + 1

    old_df = observation_df.loc[
        observation_df["TIMESTAMP"] < recent_start
    ]

    recent_df = observation_df.loc[
        observation_df["TIMESTAMP"] >= recent_start
    ]

    old_pairs = (
        old_df[
            [
                "SENDER_ACCOUNT_ID",
                "RECEIVER_ACCOUNT_ID",
            ]
        ]
        .drop_duplicates()
        .assign(seen_before=1)
    )

    recent_pairs = (
        recent_df[
            [
                "SENDER_ACCOUNT_ID",
                "RECEIVER_ACCOUNT_ID",
            ]
        ]
        .drop_duplicates()
    )

    recent_pairs = recent_pairs.merge(
        old_pairs,
        on=[
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
        ],
        how="left",
    )

    recent_pairs["is_new_counterparty"] = (
        recent_pairs["seen_before"].isna()
    ).astype(int)

    result = (
        recent_pairs
        .groupby("SENDER_ACCOUNT_ID")
        .agg(
            new_counterparties_recent=(
                "is_new_counterparty",
                "sum",
            ),
            counterparties_recent=(
                "RECEIVER_ACCOUNT_ID",
                "nunique",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    result["new_counterparty_ratio"] = (
        result["new_counterparties_recent"]
        / result["counterparties_recent"].clip(lower=1)
    )

    return result


def build_observation_features(
    observation_df: pd.DataFrame,
    observation_end: int,
    recent_windows: list[int],
    high_amount_quantile: float,
) -> pd.DataFrame:
    """Combine toutes les familles de caractéristiques."""

    outgoing = build_outgoing_features(
        observation_df=observation_df,
        observation_end=observation_end,
        recent_windows=recent_windows,
        high_amount_quantile=high_amount_quantile,
    )

    incoming = build_incoming_features(
        observation_df
    )

    network = build_network_features(
        observation_df
    )

    new_counterparties = (
        build_new_counterparty_features(
            observation_df=observation_df,
            observation_end=observation_end,
            recent_window=max(recent_windows),
        )
    )

    features = outgoing.merge(
        incoming,
        on="ACCOUNT_ID",
        how="left",
    )

    features = features.merge(
        network,
        on="ACCOUNT_ID",
        how="left",
    )

    features = features.merge(
        new_counterparties,
        on="ACCOUNT_ID",
        how="left",
    )

    features = features.fillna(0)

    features["incoming_outgoing_amount_ratio"] = safe_ratio(
        features["total_incoming_amount"],
        features["total_outgoing_amount"],
    )

    features["incoming_outgoing_tx_ratio"] = safe_ratio(
        features["n_incoming_transactions"],
        features["n_transactions"],
    )

    features["net_transaction_flow"] = (
        features["total_incoming_amount"]
        - features["total_outgoing_amount"]
    )

    features["absolute_net_flow"] = (
        features["net_transaction_flow"].abs()
    )

    return features.replace(
        [np.inf, -np.inf],
        0,
    ).fillna(0)


def build_future_target(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Crée la cible future par compte émetteur."""

    target = (
        prediction_df
        .groupby("SENDER_ACCOUNT_ID")
        .agg(
            TARGET=(
                "IS_FRAUD",
                "max",
            ),
            n_future_transactions=(
                "TX_ID",
                "count",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "SENDER_ACCOUNT_ID": "ACCOUNT_ID",
            }
        )
    )

    target["TARGET"] = target["TARGET"].astype(int)

    return target


def build_temporal_account_dataset(
    df: pd.DataFrame,
    observation_start: int,
    observation_end: int,
    prediction_start: int,
    prediction_end: int,
    recent_windows: list[int],
    high_amount_quantile: float = 0.95,
    min_history_transactions: int = 2,
) -> pd.DataFrame:
    """Construit le dataset final sans fuite temporelle."""

    observation_df, prediction_df = split_temporal_windows(
        df=df,
        observation_start=observation_start,
        observation_end=observation_end,
        prediction_start=prediction_start,
        prediction_end=prediction_end,
    )

    features = build_observation_features(
        observation_df=observation_df,
        observation_end=observation_end,
        recent_windows=recent_windows,
        high_amount_quantile=high_amount_quantile,
    )

    target = build_future_target(
        prediction_df
    )

    dataset = features.merge(
        target,
        on="ACCOUNT_ID",
        how="inner",
    )

    dataset = dataset.loc[
        dataset["n_transactions"]
        >= min_history_transactions
    ].copy()

    dataset = dataset.drop(
        columns=["n_future_transactions"]
    )

    dataset = dataset.replace(
        [np.inf, -np.inf],
        0,
    ).fillna(0)

    if dataset.empty:
        raise ValueError(
            "Le dataset temporel est vide."
        )

    return dataset.reset_index(drop=True)


def build_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build legacy account aggregates used by the public API tests.

    The training pipeline uses ``build_temporal_account_dataset`` to
    prevent future information from leaking into observation features.
    """

    ordered = df.sort_values(
        ["ACCOUNT_ID", "TIMESTAMP"]
    ).copy()
    ordered["delta_time"] = (
        ordered.groupby("ACCOUNT_ID")["TIMESTAMP"]
        .diff()
        .dt.total_seconds()
    )

    features = (
        ordered.groupby("ACCOUNT_ID")
        .agg(
            n_transactions=("AMOUNT", "count"),
            total_amount=("AMOUNT", "sum"),
            avg_amount=("AMOUNT", "mean"),
            median_amount=("AMOUNT", "median"),
            std_amount=("AMOUNT", "std"),
            min_amount=("AMOUNT", "min"),
            max_amount=("AMOUNT", "max"),
            n_counterparties=("COUNTERPARTY_ID", "nunique"),
            avg_time_between_tx=("delta_time", "mean"),
            TARGET=("IS_FRAUD", "max"),
        )
        .reset_index()
    )

    return features.fillna(0)
