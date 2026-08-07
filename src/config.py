from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
TEMPORAL_DATASET_FILE = DATA_DIR / "account_temporal_dataset.csv"

MODEL_FILE = MODEL_DIR / "fraud_model.pkl"
METADATA_FILE = MODEL_DIR / "metadata.pkl"


OBSERVATION_START = 0
OBSERVATION_END = 139

PREDICTION_START = 140
PREDICTION_END = 199


# Fenêtres utilisées pour mesurer l'activité récente.
RECENT_WINDOWS = [5, 10, 20]

# Une transaction est considérée comme élevée lorsqu'elle dépasse
# ce percentile global dans la fenêtre d'observation.
HIGH_AMOUNT_QUANTILE = 0.95

MIN_HISTORY_TRANSACTIONS = 2