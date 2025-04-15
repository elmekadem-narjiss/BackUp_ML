import pandas as pd
import logging
from influxdb_client import InfluxDBClient

# Configuration InfluxDB
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "6gVj-CNfMCVW0otLynr2-E4WTfI-ww6Z2QV0NSe-LrYfVHpFCnfGf-XUNtQ31_9CJna40ifv67fKRnKfoDnKAg=="
INFLUX_ORG = "iot_lab"
INFLUX_BUCKET = "energy_data"


# ✅ Chemin absolu vers ton dossier local
CSV_FILE_PATH = r"D:\PFE\DataSet\energy_dataset.csv"

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

        # ✅ Sauvegarde à l'emplacement spécifié
        result.to_csv(CSV_FILE_PATH, index=False)
        print(f"✅ Données exportées dans : {CSV_FILE_PATH}")
        return result

    except Exception as e:
        logging.error(f"Erreur InfluxDB : {e}")
        return pd.DataFrame()

# Exécuter
if __name__ == "__main__":
    df = load_data_from_influx()
