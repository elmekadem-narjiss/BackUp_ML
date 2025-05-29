import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database.db_connection import connect_influxdb, connect_mqtt
import pandas as pd
import logging

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie pour FastAPI."""
    print("🚀 Démarrage de l'application...")
    try:
        # Initialisation des connexions
        influx_client = connect_influxdb()
        mqtt_client = connect_mqtt()
        print("✅ Connexions InfluxDB et MQTT initialisées.")
        yield
    except Exception as e:
        print(f"❌ Erreur lors du démarrage : {e}")
        raise
    finally:
        # Nettoyage des connexions
        if 'influx_client' in locals():
            influx_client.close()
        if 'mqtt_client' in locals():
            mqtt_client.disconnect()
        print("🛑 Arrêt de l'application.")

app.lifespan = lifespan

def load_energy_consumption_data():
    """Charger les données de consommation énergétique depuis InfluxDB."""
    try:
        client = connect_influxdb()
        query = f'''
        from(bucket:"{os.getenv('INFLUX_BUCKET', 'energy_data')}")
            |> range(start:-30d)
            |> filter(fn:(r) => r._measurement == "energy")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        result = client.query_api().query_data_frame(query)
        logging.info("✅ Données chargées depuis InfluxDB.")
        return result
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement des données : {e}")
        raise

def save_data_to_influxdb(data: pd.DataFrame):
    """Sauvegarder les données dans InfluxDB."""
    try:
        client = connect_influxdb()
        write_api = client.write_api()
        write_api.write(bucket=os.getenv("INFLUX_BUCKET", "energy_data"), record=data)
        logging.info("✅ Données sauvegardées dans InfluxDB.")
    except Exception as e:
        logging.error(f"❌ Erreur lors de la sauvegarde des données : {e}")
        raise
