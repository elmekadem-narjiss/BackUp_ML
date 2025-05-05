import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from influxdb_client import InfluxDBClient
import pickle
import os

# Configuration pour InfluxDB (valeurs par défaut)
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "your-token-here"
INFLUX_ORG = "iot_lab"
INFLUX_BUCKET = "energy_data"

# Chemins pour sauvegarder le modèle et le scaler
MODEL_PATH = "model/lstm_model"
SCALER_PATH = "model/scaler.pkl"

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
    return result

def prepare_data(df):
    """Préparer les données pour l'entraînement LSTM."""
    features = ['energyproduced', 'temperature', 'humidity', 'month', 'week_of_year', 'hour']
    df = df[features].dropna()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - 30):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i:i+30])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_model():
    """Construire le modèle LSTM."""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(60, 6), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(30 * 6)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_save():
    """Entraîner et sauvegarder le modèle LSTM."""
    df = load_data_from_influx()
    X, y, scaler = prepare_data(df)
    
    model = build_model()
    history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    # Reshape output pour sauvegarder
    model.save(MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler, history, X, y

def load_model():
    """Charger le modèle et le scaler."""
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def connect_postgresql():
    """Connexion à PostgreSQL (à mocker dans les tests)."""
    import psycopg2
    return psycopg2.connect(
        dbname="energy_db",
        user="user",
        password="password",
        host="localhost",
        port="5432"
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
