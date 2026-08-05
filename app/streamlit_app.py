from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# Permet d'importer le package src lorsque Streamlit est lancé
# depuis la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.predict import predict_account  # noqa: E402


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

            XGBoost entraîné sur des caractéristiques transactionnelles
            agrégées par compte.
            """
        )

        st.write(
            """
            **Limites**

            - données utilisées à des fins de démonstration;
            - seuils de risque non validés opérationnellement;
            - prédictions dépendantes de la qualité des données;
            - validation humaine obligatoire.
            """
        )

    st.subheader("Caractéristiques du compte")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            n_transactions = st.number_input(
                "Nombre de transactions",
                min_value=1,
                value=25,
                step=1,
            )

            total_amount = st.number_input(
                "Montant total des transactions",
                min_value=0.0,
                value=75_000.0,
                step=100.0,
            )

            avg_amount = st.number_input(
                "Montant moyen",
                min_value=0.0,
                value=3_000.0,
                step=100.0,
            )

        with col2:
            median_amount = st.number_input(
                "Montant médian",
                min_value=0.0,
                value=1_500.0,
                step=100.0,
            )

            std_amount = st.number_input(
                "Écart-type des montants",
                min_value=0.0,
                value=4_200.0,
                step=100.0,
                help=(
                    "Mesure la variation des montants des transactions "
                    "autour de leur moyenne."
                ),
            )

            min_amount = st.number_input(
                "Montant minimal",
                min_value=0.0,
                value=25.0,
                step=10.0,
            )

        with col3:
            max_amount = st.number_input(
                "Montant maximal",
                min_value=0.0,
                value=25_000.0,
                step=100.0,
            )

            n_counterparties = st.number_input(
                "Nombre de contreparties distinctes",
                min_value=1,
                value=7,
                step=1,
            )

            avg_time_between_tx = st.number_input(
                "Temps moyen entre transactions, en secondes",
                min_value=0.0,
                value=1_800.0,
                step=60.0,
                help="1 800 secondes correspondent à 30 minutes.",
            )

        submitted = st.form_submit_button(
            "Analyser le compte",
            use_container_width=True,
        )

    if not submitted:
        return

    account_data = {
        "n_transactions": int(n_transactions),
        "total_amount": float(total_amount),
        "avg_amount": float(avg_amount),
        "median_amount": float(median_amount),
        "std_amount": float(std_amount),
        "min_amount": float(min_amount),
        "max_amount": float(max_amount),
        "n_counterparties": int(n_counterparties),
        "avg_time_between_tx": float(avg_time_between_tx),
    }

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

    except KeyError as error:
        st.error(
            f"Une caractéristique attendue par le modèle est absente : {error}"
        )
        return

    except Exception as error:
        st.exception(error)
        return

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

    st.subheader("Résumé des données analysées")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(
            {
                "Nombre de transactions": int(n_transactions),
                "Montant total": format_currency(total_amount),
                "Montant moyen": format_currency(avg_amount),
                "Montant médian": format_currency(median_amount),
                "Écart-type": format_currency(std_amount),
            }
        )

    with summary_col2:
        st.write(
            {
                "Montant minimal": format_currency(min_amount),
                "Montant maximal": format_currency(max_amount),
                "Contreparties distinctes": int(n_counterparties),
                "Temps moyen entre transactions": (
                    f"{avg_time_between_tx:,.0f} secondes"
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
