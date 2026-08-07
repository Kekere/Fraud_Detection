from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import (
    HIGH_AMOUNT_QUANTILE,
    METADATA_FILE,
    MIN_HISTORY_TRANSACTIONS,
    MODEL_FILE,
    OBSERVATION_END,
    OBSERVATION_START,
    PREDICTION_END,
    PREDICTION_START,
    RECENT_WINDOWS,
    TEMPORAL_DATASET_FILE,
    TRANSACTIONS_FILE,
)
from src.feature_engineering import (
    build_temporal_account_dataset,
)
from src.preprocessing import (
    clean_data,
    load_transactions,
)


def find_best_f1_threshold(
    y_true: pd.Series,
    probabilities,
) -> float:
    """Trouve le seuil maximisant le F1-score."""

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    if len(thresholds) == 0:
        return 0.5

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    return float(
        thresholds[int(f1_scores.argmax())]
    )


def fit_best_model(
    X_development: pd.DataFrame,
    y_development: pd.Series,
) -> tuple[Any, float, pd.DataFrame]:
    """Select a model using stratified out-of-fold predictions."""

    imbalance_ratio = float(
        (y_development == 0).sum()
        / (y_development == 1).sum()
    )

    candidates = [
        (
            "LogisticRegression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    C=0.2,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=600,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=400,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=2.0,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "SVM_RBF",
            make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(
                    estimator=SVC(
                        C=0.5,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        cache_size=1024,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=5,
                gamma=0.0,
                reg_alpha=0.1,
                reg_lambda=3.0,
                subsample=0.85,
                colsample_bytree=0.8,
                scale_pos_weight=1.0,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoostBalanced",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.2,
                reg_lambda=5.0,
                subsample=0.85,
                colsample_bytree=0.8,
                scale_pos_weight=imbalance_ratio ** 0.5,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = []
    splitter = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=42,
    )

    for model_name, estimator in candidates:
        probabilities = pd.Series(
            index=y_development.index,
            dtype=float,
        )
        fold_thresholds = []

        for fold_train, fold_validation in splitter.split(
            X_development,
            y_development,
        ):
            candidate = clone(estimator)
            candidate.fit(
                X_development.iloc[fold_train],
                y_development.iloc[fold_train],
            )
            fold_probabilities = candidate.predict_proba(
                X_development.iloc[fold_validation]
            )[:, 1]
            probabilities.iloc[fold_validation] = (
                fold_probabilities
            )
            fold_thresholds.append(
                find_best_f1_threshold(
                    y_development.iloc[fold_validation],
                    fold_probabilities,
                )
            )

        threshold = float(
            pd.Series(fold_thresholds).median()
        )
        predictions = (
            probabilities >= threshold
        ).astype(int)
        validation_auc = roc_auc_score(
            y_development,
            probabilities,
        )
        validation_f1 = f1_score(
            y_development,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "model": model_name,
                "roc_auc": validation_auc,
                "f1": validation_f1,
                "threshold": threshold,
            }
        )

    leaderboard = pd.DataFrame(results)
    leaderboard["selection_score"] = (
        leaderboard["roc_auc"].rank(pct=True)
        + leaderboard["f1"].rank(pct=True)
    )
    best_index = int(
        leaderboard["selection_score"].idxmax()
    )
    best_model = clone(candidates[best_index][1])
    best_model.fit(
        X_development,
        y_development,
    )

    return (
        best_model,
        float(leaderboard.loc[best_index, "threshold"]),
        leaderboard.sort_values(
            ["selection_score", "roc_auc"],
            ascending=False,
        ),
    )


def train() -> None:
    transactions = load_transactions(
        TRANSACTIONS_FILE
    )

    transactions = clean_data(
        transactions
    )

    dataset = build_temporal_account_dataset(
        df=transactions,
        observation_start=OBSERVATION_START,
        observation_end=OBSERVATION_END,
        prediction_start=PREDICTION_START,
        prediction_end=PREDICTION_END,
        recent_windows=RECENT_WINDOWS,
        high_amount_quantile=HIGH_AMOUNT_QUANTILE,
        min_history_transactions=(
            MIN_HISTORY_TRANSACTIONS
        ),
    )

    TEMPORAL_DATASET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        TEMPORAL_DATASET_FILE,
        index=False,
    )

    print(
        "\nDimensions du dataset :",
        dataset.shape,
    )

    print("\nDistribution de la cible :")
    print(dataset["TARGET"].value_counts())

    X = dataset.drop(
        columns=[
            "ACCOUNT_ID",
            "TARGET",
        ]
    )

    y = dataset["TARGET"]

    if y.nunique() < 2:
        raise ValueError(
            "La cible contient une seule classe."
        )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    X_development = pd.concat(
        [X_train, X_validation]
    ).reset_index(drop=True)
    y_development = pd.concat(
        [y_train, y_validation]
    ).reset_index(drop=True)

    model, decision_threshold, leaderboard = fit_best_model(
        X_development,
        y_development,
    )

    print("\nValidation model search:")
    print(leaderboard.to_string(index=False))

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    threshold_results = []

    diagnostic_thresholds = sorted(
        {
            round(
                min(max(decision_threshold * factor, 0.01), 0.99),
                4,
            )
            for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        }
    )

    for threshold in diagnostic_thresholds:
        threshold_predictions = (
            test_probabilities >= threshold
        ).astype(int)

        threshold_results.append(
            {
                "threshold": threshold,
                "alerts": int(
                    threshold_predictions.sum()
                ),
                "precision": precision_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0,
                ),
            }
        )

    threshold_table = pd.DataFrame(
        threshold_results
    )

    print("\nComparaison des seuils :")
    print(threshold_table)

    test_predictions = (
        test_probabilities >= decision_threshold
    ).astype(int)

    print(
        "\nSeuil de décision :",
        round(decision_threshold, 4),
    )

    print("\nRapport de classification :")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    print("\nMatrice de confusion :")
    print(
        confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    average_precision = average_precision_score(
        y_test,
        test_probabilities,
    )

    print(
        "\nROC-AUC :",
        round(roc_auc, 4),
    )

    print(
        "Average precision :",
        round(average_precision, 4),
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "algorithm": str(leaderboard.iloc[0]["model"]),
        "features": list(X.columns),
        "target": "TARGET",
        "decision_threshold": decision_threshold,
        "observation_start": OBSERVATION_START,
        "observation_end": OBSERVATION_END,
        "prediction_start": PREDICTION_START,
        "prediction_end": PREDICTION_END,
        "recent_windows": RECENT_WINDOWS,
        "high_amount_quantile": (
            HIGH_AMOUNT_QUANTILE
        ),
        "validation_search": leaderboard.to_dict(
            orient="records"
        ),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "training_accounts": len(X_train),
        "test_accounts": len(X_test),
    }

    joblib.dump(
        metadata,
        METADATA_FILE,
    )

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(
        model[-1],
        "coef_",
    ):
        importance_values = abs(model[-1].coef_[0])
    else:
        importance_values = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        ).importances_mean

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": importance_values,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\nTop 25 caractéristiques :")
    print(
        feature_importance.head(25)
    )

    print(
        f"\nModèle sauvegardé dans : "
        f"{MODEL_FILE}"
    )


if __name__ == "__main__":
    train()
