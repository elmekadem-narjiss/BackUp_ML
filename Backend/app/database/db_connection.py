import redis
import psycopg2
from influxdb_client import InfluxDBClient
import paho.mqtt.client as mqtt
import os

# Configuration pour InfluxDB
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "6gVj-CNfMCVW0otLynr2-E4WTfI-ww6Z2QV0NSe-LrYfVHpFCnfGf-XUNtQ31_9CJna40ifv67fKRnKfoDnKAg==")
INFLUX_ORG = os.getenv("INFLUX_ORG", "iot_lab")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "energy_data")

# Configuration pour MQTT
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "energy/data")

# Configuration pour Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Configuration pour PostgreSQL
PG_DBNAME = os.getenv("PG_DBNAME", "energy_db")
PG_USER = os.getenv("PG_USER", "user")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")

def connect_influxdb():
    """Établir une connexion à InfluxDB."""
    print("🌐 Tentative de connexion à InfluxDB...")
    print(f"🔍 Vérification des variables :")
    print(f"  - INFLUX_URL: {INFLUX_URL}")
    print(f"  - INFLUX_TOKEN: {'Présent' if INFLUX_TOKEN else 'Absent'}")
    print(f"  - INFLUX_ORG: {INFLUX_ORG}")
    print(f"  - INFLUX_BUCKET: {INFLUX_BUCKET}")
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        print("✅ Connexion à InfluxDB établie.")
        return client
    except Exception as e:
        print(f"⚠️ Erreur lors de la connexion à InfluxDB : {e}")
        raise

def connect_mqtt():
    """Établir une connexion au broker MQTT."""
    try:
        client = mqtt.Client(client_id="", protocol=mqtt.MQTTv5)
        client.connect(MQTT_BROKER, MQTT_PORT)
        print(f"✅ Connecté au broker MQTT à {MQTT_BROKER}:{MQTT_PORT}")
        return client
    except Exception as e:
        print(f"❌ Erreur lors de la connexion à MQTT : {e}")
        raise

def connect_redis():
    """Établir une connexion à Redis."""
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        print(f"✅ Connecté à Redis à {REDIS_HOST}:{REDIS_PORT}")
        return client
    except Exception as e:
        print(f"❌ Erreur lors de la connexion à Redis : {e}")
        raise

def connect_postgresql():
    """Établir une connexion à PostgreSQL."""
    try:
        conn = psycopg2.connect(
            dbname=PG_DBNAME,
            user=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT
        )
        print(f"✅ Connecté à PostgreSQL à {PG_HOST}:{PG_PORT}")
        return conn
    except Exception as e:
        print(f"❌ Erreur lors de la connexion à PostgreSQL : {e}")
        raise
