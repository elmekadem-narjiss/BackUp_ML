from tensorflow.keras.models import load_model as keras_load_model
import pickle
import os
import logging

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration pour InfluxDB
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv(
    "INFLUX_TOKEN",
    "6gVj-CNfMCVW0otLynr2-E4WTfI-ww6Z2QV0NSe-LrYfVHpFCnfGf-XUNtQ31_9CJna40ifv67fKRnKfoDnKAg=="
)
INFLUX_ORG = os.getenv("INFLUX_ORG", "iot_lab")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "energy_data")

# Configuration du modèle LSTM
MODEL_PATH = "model/lstm_model.keras"
SCALER_PATH = "model/scaler.pkl"
SEQ_LENGTH = 60
PREDICTION_DAYS = 30
FEATURES = ['energyproduced', 'temperature', 'humidity', 'month', 'week_of_year', 'hour']

# Initialisation du modèle et du scaler
MODEL = None
SCALER = None

def load_model():
    """Charger le modèle et le scaler."""
    try:
        logger.debug("Chargement du modèle depuis %s", MODEL_PATH)
        model = keras_load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logger.debug("Modèle et scaler chargés avec succès")
        return model, scaler
    except Exception as e:
        logger.error("Erreur lors du chargement du modèle ou scaler : %s", e)
        raise
