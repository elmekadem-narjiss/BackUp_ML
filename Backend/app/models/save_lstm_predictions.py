import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import logging
from datetime import datetime
import psycopg2
from fastapi import HTTPException
from app.config.config import PG_DBNAME, PG_USER,PG_PASSWORD,PG_HOST,PG_PORT


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

def save_predictions_to_db(predictions_df):
    try:
        conn = connect_postgresql()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO predictions_lstm (
            energyproduced, temperature, humidity, month, week_of_year, hour, prediction_day
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        current_timestamp = datetime.now()

        for _, row in predictions_df.iterrows():
            cursor.execute(insert_query, (
                float(row['energyproduced']),
                float(row['temperature']),
                float(row['humidity']),
                int(row['month']),
                int(row['week_of_year']),
                int(row['hour']),
                current_timestamp
            ))

        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"{len(predictions_df)} prédictions enregistrées dans la base de données.")
    except Exception as e:
        logging.error(f"Erreur lors de l'enregistrement des prédictions : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement des prédictions : {e}")

def main():
    try:
        # Charger le fichier CSV
        predictions_df = pd.read_csv('D:/PFE/DataSet/final_lstm_predictions.csv')

        # S'assurer que les colonnes temporelles sont bien des entiers
        predictions_df['month'] = predictions_df['month'].astype(int)
        predictions_df['week_of_year'] = predictions_df['week_of_year'].astype(int)
        predictions_df['hour'] = predictions_df['hour'].astype(int)

        save_predictions_to_db(predictions_df)

    except Exception as e:
        logging.error(f"Erreur lors de la lecture ou de l'insertion : {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
#https://gitlab.com/pfe8425216/backend_ml

#git init --initial-branch=main
#git remote add origin git@gitlab.com:pfe8425216/Backend_ML.git
#git add .
#git commit -m "Initial commit"
#git push --set-upstream origin main