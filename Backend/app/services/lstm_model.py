import tensorflow as tf
import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from keras.api.models import Model
from keras.api.layers import Dense, LSTM, Input, RepeatVector, TimeDistributed
from keras.api.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from influxdb_client import InfluxDBClient
from fastapi.responses import JSONResponse

# InfluxDB params
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "6gVj-CNfMCVW0otLynr2-E4WTfI-ww6Z2QV0NSe-LrYfVHpFCnfGf-XUNtQ31_9CJna40ifv67fKRnKfoDnKAg=="
INFLUX_ORG = "iot_lab"
INFLUX_BUCKET = "energy_data"

# LSTM params
SEQ_LENGTH = 60
PREDICTION_DAYS = 30

# Load data from InfluxDB
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
          |> keep(columns: ["_time", "energyproduced", "temperature", "humidity", "month", "week_of_year", "hour"])
        '''

        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        result = query_api.query_data_frame(org=INFLUX_ORG, query=query)
        client.close()

        result = result.dropna()
        return result
    except Exception as e:
        logging.error(f"Erreur InfluxDB : {e}")
        return pd.DataFrame()

# Prepare LSTM data
def prepare_data(df):
    features = ['energyproduced', 'temperature', 'humidity', 'month', 'week_of_year', 'hour']
    df = df[features]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - SEQ_LENGTH - PREDICTION_DAYS):
        X.append(df_scaled[i:i + SEQ_LENGTH])
        y.append(df_scaled[i + SEQ_LENGTH:i + SEQ_LENGTH + PREDICTION_DAYS])  # toutes les features

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Build LSTM model
def build_model():
    input_layer = Input(shape=(SEQ_LENGTH, 6))
    x = LSTM(64, activation='relu', return_sequences=True)(input_layer)
    x = LSTM(64, activation='relu')(x)
    x = RepeatVector(PREDICTION_DAYS)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    output = TimeDistributed(Dense(6))(x)  # prédiction de 6 features

    model = Model(input_layer, output)
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

# Train and save model
def train_and_save():
    df = load_data_from_influx()
    if df.empty:
        logging.error("Pas de données")
        return None, None

    X, y, scaler = prepare_data(df)
    model = build_model()
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)

    model.save("lstm_encoder_decoder_multi.h5")
    with open("scaler_multi.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler

# Load model
def load_model():
    try:
        model = tf.keras.models.load_model("lstm_encoder_decoder_multi.h5")
        with open("scaler_multi.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        logging.warning("Modèle non trouvé, entraînement requis")
        return train_and_save()

# Load or train at start
MODEL, SCALER = load_model()
# Toujours réentraîner le modèle à chaque démarrage
#MODEL, SCALER = train_and_save()



from app.config.config import  PG_DBNAME, PG_USER, PG_PASSWORD, PG_HOST, PG_PORT
from fastapi import  HTTPException
import psycopg2
from datetime import datetime

def connect_postgresql():
    """Connexion à PostgreSQL"""
    try:
        logging.debug("Tentative de connexion à PostgreSQL...")
        conn = psycopg2.connect(
            dbname=PG_DBNAME, user=PG_USER, password=PG_PASSWORD, host=PG_HOST, port=PG_PORT
        )
        logging.info("Connexion à PostgreSQL réussie.")
        return conn
    except Exception as e:
        logging.error(f"Erreur PostgreSQL : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur PostgreSQL : {e}")


# Fonction pour enregistrer les prédictions dans la base de données PostgreSQL
def save_predictions_to_db(predictions):
    try:
        conn = connect_postgresql()
        cursor = conn.cursor()

        # Requête d'insertion dans la nouvelle table lstm_predictions
        insert_query = """
        INSERT INTO lstm_predictions (energyproduced, temperature, humidity, month, week_of_year, hour, prediction_day)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
       # Récupérer la date et l'heure actuelles
        current_timestamp = datetime.now()
        
        for prediction in predictions:
            cursor.execute(insert_query, (
                prediction['energyproduced'],
                prediction['temperature'],
                prediction['humidity'],
                prediction['month'],
                prediction['week_of_year'],
                prediction['hour'],
                current_timestamp 
            ))

        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"{len(predictions)} prédictions enregistrées dans la base de données.")
    except Exception as e:
        logging.error(f"Erreur lors de l'enregistrement des prédictions : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement des prédictions : {e}")


#######################################  MODEL Entrainess avec toutes les donnes ######################
    

    # Installer TensorFlow si nécessaire
#!pip install tensorflow

#from google.colab import drive
#drive.mount('/content/drive')

import tensorflow as tf
import logging
import pickle
import numpy as np
import pandas as pd
#from keras.models import Model
#from keras.layers import Dense, LSTM, Input, RepeatVector, TimeDistributed, Dropout, Bidirectional, Attention
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
#from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime

# Paramètres
SEQ_LENGTH = 60
PREDICTION_DAYS = 30
TRAIN_SIZE = 999
CSV_FILE_PATH = '/content/drive/MyDrive/energy_dataset.csv'

# Fonction de perte SMAPE
def smape_loss(y_true, y_pred):
    numerator = tf.abs(y_true - y_pred)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0
    return tf.reduce_mean(numerator / (denominator + 1e-7))

# Chargement et prétraitement des données
def load_data_from_csv():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df.dropna(inplace=True)

        # Ajout de bruit pour l'augmentation de données
        df['energyproduced'] += np.random.normal(0, 0.01, size=len(df))
        df['temperature'] += np.random.normal(0, 0.1, size=len(df))

        df['month'] = df['month'].astype(int)
        df['week_of_year'] = df['week_of_year'].astype(int)
        df['hour'] = df['hour'].astype(int)

        # Nouvelles features
        df['is_weekend'] = df['week_of_year'].apply(lambda x: 1 if x % 7 in [0, 6] else 0)
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

        return df
    except Exception as e:
        logging.error(f"Erreur de chargement du fichier CSV : {e}")
        return pd.DataFrame()

# Préparation des données
def prepare_data(df):
    features = [
        'energyproduced', 'temperature', 'humidity',
        'month', 'week_of_year', 'hour',
        'is_weekend', 'sin_hour', 'cos_hour',
        'sin_month', 'cos_month'
    ]

    df = df[features].head(TRAIN_SIZE + SEQ_LENGTH + PREDICTION_DAYS)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - SEQ_LENGTH - PREDICTION_DAYS):
        X.append(df_scaled[i:i + SEQ_LENGTH])
        y.append(df_scaled[i + SEQ_LENGTH:i + SEQ_LENGTH + PREDICTION_DAYS])

    return np.array(X), np.array(y), scaler

# Modèle LSTM avec Attention
def build_model(input_dim):
    input_layer = Input(shape=(SEQ_LENGTH, input_dim))

  #  x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
   # x = Dropout(0.3)(x)
   # x = Bidirectional(LSTM(64, return_sequences=False))(x)
   # x = Dropout(0.3)(x)

    x = RepeatVector(PREDICTION_DAYS)(x)
    x = LSTM(64, return_sequences=True)(x)
   # x = Attention()([x, x])
    x = LSTM(32, return_sequences=True)(x)

    output = TimeDistributed(Dense(input_dim))(x)

    model = Model(inputs=input_layer, outputs=output)
    #model.compile(optimizer=Adam(0.001), loss=smape_loss)
    return model

# Entraînement + callbacks + visualisation
def train_and_save():
    df = load_data_from_csv()
    if df.empty:
        logging.error("Pas de données")
        return None, None

    X, y, scaler = prepare_data(df)
    input_dim = X.shape[2]
    model = build_model(input_dim)

    log_dir = "/content/drive/MyDrive/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
       # EarlyStopping(patience=10, restore_best_weights=True),
       # ReduceLROnPlateau(factor=0.2, patience=5),
       # TensorBoard(log_dir=log_dir)
    ]

    history = model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=callbacks
    )

    model.save('/content/drive/MyDrive/lstm_with_attention.h5')
    with open('/content/drive/MyDrive/scaler_with_attention.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler, history, X, y

# Lancer l'entraînement
model, scaler, history, X, y = train_and_save()

# Courbe d'apprentissage
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Courbe d\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('SMAPE Loss')
plt.legend()
plt.grid(True)
plt.show()

# Visualisation Prédiction vs Réel
def plot_prediction(X, y_true, y_pred, scaler):
    index = np.random.randint(0, len(X))
    true = scaler.inverse_transform(y_true[index])
    pred = scaler.inverse_transform(y_pred[index])

    plt.figure(figsize=(12, 5))
    for i in range(true.shape[1]):
        plt.plot(true[:, i], label=f'True {i}')
        plt.plot(pred[:, i], linestyle='--', label=f'Pred {i}')
    plt.legend()
    plt.title('Comparaison Prédiction vs Réel')
    plt.grid(True)
    plt.show()

# Afficher une prédiction
y_pred = model.predict(X)
plot_prediction(X, y, y_pred, scaler)

# Lancer TensorBoard si besoin :
# %load_ext tensorboard
# %tensorboard --logdir /content/drive/MyDrive/logs
