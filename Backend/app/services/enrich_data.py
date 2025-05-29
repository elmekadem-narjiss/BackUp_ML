import pandas as pd
from sqlalchemy import create_engine
from influxdb_client import InfluxDBClient
import os

# Configuration pour PostgreSQL
PG_DBNAME = os.getenv("PG_DBNAME", "energy_db")
PG_USER = os.getenv("PG_USER", "user")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")

# Configuration pour InfluxDB
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "6gVj-CNfMCVW0otLynr2-E4WTfI-ww6Z2QV0NSe-LrYfVHpFCnfGf-XUNtQ31_9CJna40ifv67fKRnKfoDnKAg==")
INFLUX_ORG = os.getenv("INFLUX_ORG", "iot_lab")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "energy_data")

def add_time_features(df):
    """Ajouter des caractéristiques temporelles au DataFrame."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['hour'] = df['timestamp'].dt.hour
    return df

def load_data_from_postgres():
    """Charger les données depuis PostgreSQL."""
    engine = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}")
    query = "SELECT * FROM energy_data"
    df = pd.read_sql(query, engine)
    return df

def save_to_influx(data: pd.DataFrame):
    """Sauvegarder les données dans InfluxDB."""
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api()
    write_api.write(bucket=INFLUX_BUCKET, record=data)
    client.close()

def query_influx():
    """Interroger les données depuis InfluxDB."""
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
