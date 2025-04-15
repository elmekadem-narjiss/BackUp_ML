import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import logging
import pandas as pd
import mlflow
import mlflow.statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from influxdb_client import InfluxDBClient
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

#just add some code 
# === CONFIGURATION LOGGING ===
logger = logging.getLogger("EnergyForecast")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# === CONFIGURATION MLFlow ===
mlflow.set_tracking_uri("http://localhost:5000")  # URL de suivi MLflow
mlflow.set_experiment("SARIMAX_EnergyConsumption")  # Nom de l'expérience dans MLflow
mlflow.autolog()  # Activation automatique du suivi des modèles et métriques

# === IMPORT DES VARIABLES DE CONFIGURATION ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
from app.config.config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG

# === RÉCUPÉRATION DES DONNÉES DE InfluxDB ===
def get_data_from_influx():
    logger.info("Connexion à InfluxDB...")
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN)
    query_api = client.query_api()

    query = """
    from(bucket: "energy_data")
    |> range(start: 0)
    |> filter(fn: (r) => r._measurement == "environment_data")
    |> filter(fn: (r) => r._field == "energyConsumption")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    try:
        result = query_api.query_data_frame(query, org=INFLUX_ORG)
        result["_time"] = pd.to_datetime(result["_time"]).dt.tz_localize(None)
        result.set_index("_time", inplace=True)
        result = result.asfreq('D')
        logger.info("Données chargées avec succès depuis InfluxDB.")
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données InfluxDB: {e}")
        sys.exit(1)
    finally:
        client.close()

    return result

# === NETTOYAGE DES DONNÉES ===
def clean_missing_data(series):
    series = series.ffill().bfill()
    if series.isna().any():
        default = series.mean()
        if pd.isna(default):
            default = 0
        series = series.fillna(default)
    return series

def split_and_clean_data(df):
    logger.info("Vérification des NaN dans les données brutes:")
    logger.info(df.isna().sum())

    train = df.iloc[:-24]
    test = df.iloc[-24:]

    train.loc[:, "energyConsumption"] = clean_missing_data(train["energyConsumption"])
    test.loc[:, "energyConsumption"] = test["energyConsumption"].fillna(train["energyConsumption"].mean())

    logger.info("Données nettoyées (NaN traités).")
    return train, test

# === ENTRAÎNEMENT DU MODÈLE SARIMAX ===
def train_sarimax_model(train_series, order, seasonal_order):
    logger.info(f"Entraînement du modèle SARIMAX avec order={order}, seasonal_order={seasonal_order}...")
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)

# === GÉNÉRATION DU GRAPHIQUE DES PRÉVISIONS ===
def plot_forecast(train, test, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train["energyConsumption"], label="Train")
    plt.plot(test.index, test["energyConsumption"], label="Test")
    plt.plot(forecast.index, forecast, label="Prévision", linestyle="--")
    plt.title("Prévision SARIMAX - Consommation énergétique")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("forecast_plot.png")
    mlflow.log_artifact("forecast_plot.png")  # Sauvegarde comme artefact dans MLflow
    logger.info("Graphique des prévisions sauvegardé et ajouté comme artefact.")

# === FONCTION PRINCIPALE ===
def run():
    df = get_data_from_influx()  # Récupérer les données depuis InfluxDB
    train, test = split_and_clean_data(df)  # Nettoyage et découpage des données

    with mlflow.start_run():  # Démarrer un suivi d'expérience avec MLflow
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)

        mlflow.log_param("order", order)  # Enregistrer les paramètres du modèle dans MLflow
        mlflow.log_param("seasonal_order", seasonal_order)

        model_fit = train_sarimax_model(train["energyConsumption"], order, seasonal_order)  # Entraînement du modèle

        input_example = pd.DataFrame({
            'start': [test.index[0]],
            'end': [test.index[-1]],
            'energyConsumption': [train["energyConsumption"].iloc[-1]]
        })

        signature = infer_signature(input_example, model_fit.predict(start=test.index[0], end=test.index[-1]))  # Inférer la signature

        mlflow.statsmodels.log_model(
            model_fit,
            artifact_path="sarimax_model",  # Sauvegarde du modèle
            signature=signature
        )

        # Enregistrer les métriques dans MLflow
        mlflow.log_metric("aic", model_fit.aic)
        mlflow.log_metric("bic", model_fit.bic)

        forecast = model_fit.predict(start=test.index[0], end=test.index[-1])
        valid_idx = test["energyConsumption"].notna() & forecast.notna()

        if valid_idx.sum() > 0:
            mse = mean_squared_error(test["energyConsumption"][valid_idx], forecast[valid_idx])
            mlflow.log_metric("mse", mse)  # Enregistrer le MSE
            logger.info(f"✅ MSE : {mse:.4f}")
        else:
            logger.warning("⚠️ Aucune donnée valide pour le calcul du MSE.")

        plot_forecast(train, test, forecast)  # Tracer et enregistrer les prévisions

        # LangChain remplacé par une réponse simulée (solution custom)
        answer = "MLflow is an open-source platform for managing the ML lifecycle including experimentation, reproducibility, and deployment."
        logger.info(f"(Réponse simulée) LangChain Answer: {answer}")

# === POINT D'ENTRÉE ===
if __name__ == "__main__":
    run()
