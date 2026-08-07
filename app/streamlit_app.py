from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permet d'importer le package src lorsque Streamlit est lancé
# depuis la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import TEMPORAL_DATASET_FILE
from src.explain import explain_account
from src.predict import predict_account

st.set_page_config(
    page_title="Détection de comptes à risque",
    page_icon="🔎",
    layout="wide",
)


def format_currency(value: float) -> str:
    """Formate une valeur monétaire pour l'affichage."""

    return f"{value:,.2f} $".replace(",", " ")


def get_risk_level(risk_score: float) -> tuple[str, str]:
    """
    Retourne le niveau de risque et une recommandation.

    Ces seuils servent uniquement à la démonstration.
    Ils devraient être validés avec les parties prenantes métier.
    """

    if risk_score >= 70:
        return (
            "Élevé",
            "Prioriser ce compte pour une analyse approfondie.",
        )

    if risk_score >= 40:
        return (
            "Modéré",
            "Examiner les facteurs de risque avant de décider d'une intervention.",
        )

    return (
        "Faible",
        "Aucune intervention prioritaire n'est recommandée à ce stade.",
    )


@st.cache_data
def load_account_features() -> pd.DataFrame:
    """Charge les caractéristiques produites par le pipeline temporel."""

    return pd.read_csv(TEMPORAL_DATASET_FILE)


def main() -> None:
    st.title("Détection de comptes présentant un risque de fraude")

    st.write(
        """
        Ce prototype estime la probabilité qu'un compte présente un
        comportement atypique à partir de caractéristiques agrégées de
        ses transactions.
        """
    )

    st.info(
        """
        Le score généré constitue une aide à la priorisation des dossiers.
        Il ne constitue ni une preuve de fraude ni une décision automatisée.
        Toute intervention doit être validée par un analyste.
        """
    )

    with st.sidebar:
        st.header("À propos")

        st.write(
            """
            **Objectif**

            Aider les analystes à prioriser les comptes nécessitant une
            vérification supplémentaire.
            """
        )

        st.write(
            """
            **Modèle**

            Extra Trees entraîné sur des caractéristiques transactionnelles
            agrégées par compte.
            """
        )


    st.subheader("Sélection du compte")

    try:
        dataset = load_account_features()
    except FileNotFoundError:
        st.error(
            "Le dataset de caractéristiques est introuvable. Exécutez "
            "`python -m src.train` pour le générer."
        )
        return

    account_ids = dataset["ACCOUNT_ID"].tolist()

    with st.form("prediction_form"):
        selected_account_id = st.selectbox(
            "Identifiant du compte",
            options=account_ids,
        )
        submitted = st.form_submit_button(
            "Analyser le compte",
            use_container_width=True,
        )

    if not submitted:
        return

    account_row = dataset.loc[
        dataset["ACCOUNT_ID"] == selected_account_id
    ].iloc[0]
    account_data = account_row.drop(
        labels=["ACCOUNT_ID", "TARGET"],
        errors="ignore",
    ).to_dict()

    try:
        result = predict_account(account_data)

    except FileNotFoundError:
        st.error(
            """
            Le fichier du modèle est introuvable. Exécute d'abord
            `python -m src.train` afin de générer le modèle.
            """
        )
        return

    except (KeyError, ValueError) as error:
        st.error(
            f"Une caractéristique attendue par le modèle est absente : {error}"
        )
        return

    except (OSError, TypeError, AttributeError, IndexError) as error:
        st.exception(error)
        return

    try:
        explanation = explain_account(account_data)
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
    ) as error:
        explanation = None
        explanation_error = error

    prediction = result["prediction"]
    risk_score = float(result["risk_score"])

    risk_level, recommendation = get_risk_level(risk_score)

    st.divider()
    st.subheader("Résultat de l'analyse")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            label="Score de risque",
            value=f"{risk_score:.2f} %",
        )

    with metric_col2:
        st.metric(
            label="Niveau de risque",
            value=risk_level,
        )

    with metric_col3:
        st.metric(
            label="Classe prédite",
            value=(
                "Compte à risque"
                if prediction == 1
                else "Compte non prioritaire"
            ),
        )

    st.progress(
        min(max(risk_score / 100, 0.0), 1.0)
    )

    if risk_level == "Élevé":
        st.error(recommendation)

    elif risk_level == "Modéré":
        st.warning(recommendation)

    else:
        st.success(recommendation)

    st.subheader("Explicabilité de la prédiction")

    if explanation is None:
        st.warning(
            "La prédiction a réussi, mais son explication n'a pas pu "
            f"être générée : {explanation_error}"
        )
    else:
        contributions = explanation["contributions"].head(10).copy()
        contributions["direction"] = contributions["shap_value"].apply(
            lambda value: (
                "Augmente le risque"
                if value > 0
                else "Réduit le risque"
            )
        )

        st.caption(
            "Une contribution SHAP positive pousse la prédiction vers "
            "la classe fraude; une contribution négative l'en éloigne."
        )

        chart_data = contributions.set_index("feature")[["shap_value"]]
        st.bar_chart(
            chart_data,
            horizontal=True,
            use_container_width=True,
        )

        st.dataframe(
            contributions[
                [
                    "feature",
                    "value",
                    "shap_value",
                    "direction",
                ]
            ].rename(
                columns={
                    "feature": "Variable",
                    "value": "Valeur",
                    "shap_value": "Contribution SHAP",
                    "direction": "Effet",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Résumé des données analysées")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(
            {
                "Identifiant": selected_account_id,
                "Nombre de transactions": int(
                    account_data["n_transactions"]
                ),
                "Montant total sortant": format_currency(
                    account_data["total_outgoing_amount"]
                ),
                "Montant moyen sortant": format_currency(
                    account_data["avg_outgoing_amount"]
                ),
                "Contreparties": int(
                    account_data["n_counterparties"]
                ),
            }
        )

    with summary_col2:
        st.write(
            {
                "Transactions entrantes": int(
                    account_data["n_incoming_transactions"]
                ),
                "Montant total entrant": format_currency(
                    account_data["total_incoming_amount"]
                ),
                "Degré total du réseau": int(
                    account_data["total_degree"]
                ),
                "Temps moyen entre transactions": (
                    f"{account_data['avg_steps_between_tx']:,.1f} étapes"
                ),
            }
        )

    st.caption(
        """
        Les seuils faible, modéré et élevé sont définis uniquement pour ce
        prototype. Dans une solution réelle, ils devraient être établis à
        partir des coûts d'enquête, de la capacité opérationnelle, du taux de
        faux positifs acceptable et des objectifs de rappel du modèle.
        """
    )


if __name__ == "__main__":
    main()
