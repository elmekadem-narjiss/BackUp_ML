import os
import sys
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
from keras.api.models import Model
from keras.api.layers import Dense, LSTM, Input, RepeatVector, TimeDistributed
from keras.api.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from influxdb_client import InfluxDBClient
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature

# === CONFIGURATION LOGGING ===
logger = logging.getLogger("LSTM_Energy")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# === MLflow Configuration ===
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LSTM_EnergyForecast")
mlflow.tensorflow.autolog()

# === InfluxDB Configuration ===
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "6gVj-CNfMCVW0otLynr2-E4WTfI-ww6Z2QV0NSe-LrYfVHpFCnfGf-XUNtQ31_9CJna40ifv67fKRnKfoDnKAg=="
INFLUX_ORG = "iot_lab"
INFLUX_BUCKET = "energy_data"

# === LSTM Params ===
SEQ_LENGTH = 60
PREDICTION_DAYS = 30

# === Charger les données depuis InfluxDB ===
def load_data_from_influx():
    try:
        query = '''
        from(bucket: "energy_data")
          |> range(start: 0)
          |> filter(fn: (r) => r["_measurement"] == "energy_data")
          |> filter(fn: (r) =>
              r["_field"] == "energyproduced" or
              r["_field"] == "temperature" or
              r["_field"] == "humidity" or
              r["_field"] == "month" or
              r["_field"] == "week_of_year" or
              r["_field"] == "hour")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        result = query_api.query_data_frame(org=INFLUX_ORG, query=query)
        client.close()

        result = result.dropna()
        result["_time"] = pd.to_datetime(result["_time"])
        result.set_index("_time", inplace=True)
        return result
    except Exception as e:
        logger.error(f"Erreur InfluxDB : {e}")
        return pd.DataFrame()

# === Préparer les données pour LSTM ===
def prepare_data(df):
    features = ['energyproduced', 'temperature', 'humidity', 'month', 'week_of_year', 'hour']
    df = df[features]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - SEQ_LENGTH - PREDICTION_DAYS):
        X.append(df_scaled[i:i + SEQ_LENGTH])
        y.append(df_scaled[i + SEQ_LENGTH:i + SEQ_LENGTH + PREDICTION_DAYS])

    return np.array(X), np.array(y), scaler

# === Construire le modèle LSTM ===
def build_model():
    input_layer = Input(shape=(SEQ_LENGTH, 6))
    x = LSTM(64, activation='relu', return_sequences=True)(input_layer)
    x = LSTM(64, activation='relu')(x)
    x = RepeatVector(PREDICTION_DAYS)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    output = TimeDistributed(Dense(6))(x)

    model = Model(input_layer, output)
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

# === Tracer et log la prévision ===
def plot_prediction(prediction, title="Prévision LSTM"):
    plt.figure(figsize=(12, 6))
    for i in range(prediction.shape[2]):
        plt.plot(prediction[0, :, i], label=f'feature_{i}')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("lstm_forecast.png")
    mlflow.log_artifact("lstm_forecast.png")
    logger.info("✅ Graphe des prévisions loggé dans MLflow")

# === Exécution principale ===
def run():
    df = load_data_from_influx()
    if df.empty:
        logger.error("Pas de données chargées.")
        return

    X, y, scaler = prepare_data(df)
    model = build_model()

    with mlflow.start_run():
        logger.info("🔁 Entraînement du modèle LSTM en cours...")
        history = model.fit(X, y, epochs=10, batch_size=16, verbose=1)

        logger.info("✅ Modèle entraîné. Sauvegarde...")
        model.save("lstm_model.h5")
        mlflow.log_artifact("lstm_model.h5")

        # Inference sur les dernières données
        prediction = model.predict(X[-1:].reshape(1, SEQ_LENGTH, 6))

        # Log signature et métriques
        input_example = X[-1:].reshape(1, SEQ_LENGTH, 6)
        output_example = prediction
        signature = infer_signature(input_example, output_example)
        mlflow.keras.log_model(model, "lstm_model", signature=signature)

        mse = mean_squared_error(y[-1].flatten(), prediction.flatten())
        mlflow.log_metric("mse", mse)
        logger.info(f"📉 MSE: {mse:.4f}")

        # Plot
        plot_prediction(prediction)

        # Evaluation du modèle
        logger.info("📊 Evaluation du modèle...")
        evaluation = model.evaluate(X, y)
        logger.info(f"📊 Evaluation: {evaluation:.4f}")

# === Point d'entrée ===
if __name__ == "__main__":
    run()


#just add some test