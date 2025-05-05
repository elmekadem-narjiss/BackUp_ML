import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from influxdb_client import InfluxDBClient
import pickle
import os

# Configuration pour InfluxDB
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "6gVj-CNfMCVW0otLynr2-E4WTfI-ww6Z2QV0NSe-LrYfVHpFCnfGf-XUNtQ31_9CJna40ifv67fKRnKfoDnKAg==")
INFLUX_ORG = os.getenv("INFLUX_ORG", "iot_lab")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "energy_data")

# Configuration du modèle LSTM
MODEL_PATH = "model/lstm_model"
SCALER_PATH = "model/scaler.pkl"
SEQ_LENGTH = 60
PREDICTION_DAYS = 30
FEATURES = ['energyproduced', 'temperature', 'humidity', 'month', 'week_of_year', 'hour']

# Initialisation du modèle et du scaler
MODEL = None
SCALER = None

def initialize_model_and_scaler():
    """Initialiser le modèle et le scaler si nécessaire."""
    global MODEL, SCALER
    if MODEL is None or SCALER is None:
        try:
            MODEL, SCALER = load_model()
        except FileNotFoundError:
            MODEL = build_model()
            SCALER = MinMaxScaler()

def load_data_from_influx():
    """Charger les données depuis InfluxDB."""
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query = f'''
    from(bucket:"{INFLUX_BUCKET}")
        |> range(start:-30d)
        |> filter(fn:(r) => r._measurement == "energy")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    result = client.query_api().query_data_frame(query)
    client.close()
    return result

def prepare_data(df):
    """Préparer les données pour l'entraînement LSTM."""
    df = df[FEATURES].dropna()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(SEQ_LENGTH, len(scaled_data) - PREDICTION_DAYS):
        X.append(scaled_data[i-SEQ_LENGTH:i])
        y.append(scaled_data[i:i+PREDICTION_DAYS])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_model():
    """Construire le modèle LSTM."""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, len(FEATURES)), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(PREDICTION_DAYS * len(FEATURES))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_save():
    """Entraîner et sauvegarder le modèle LSTM."""
    df = load_data_from_influx()
    X, y, scaler = prepare_data(df)
    
    model = build_model()
    history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    model.save(MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    global MODEL, SCALER
    MODEL = model
    SCALER = scaler
    
    return model, scaler, history, X, y

def load_model():
    """Charger le modèle et le scaler."""
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def connect_postgresql():
    """Connexion à PostgreSQL."""
    import psycopg2
    return psycopg2.connect(
        dbname=os.getenv("PG_DBNAME", "energy_db"),
        user=os.getenv("PG_USER", "user"),
        password=os.getenv("PG_PASSWORD", "password"),
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432")
    )

def save_predictions_to_db(predictions):
    """Sauvegarder les prédictions dans PostgreSQL."""
    conn = connect_postgresql()
    cursor = conn.cursor()
    for pred in predictions:
        cursor.execute(
            """
            INSERT INTO predictions (
                energyproduced, temperature, humidity, month, week_of_year, hour
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                pred['energyproduced'], pred['temperature'], pred['humidity'],
                pred['month'], pred['week_of_year'], pred['hour']
            )
        )
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Exécuter l'entraînement uniquement si le script est appelé directement
    model, scaler, history, X, y = train_and_save()
    print("Entraînement terminé.")
