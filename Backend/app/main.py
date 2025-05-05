from fastapi import FastAPI
from app.utils.time_series import load_energy_consumption_data, save_data_to_influxdb
from app.services.prediction_service import apply_arima_model, save_predictions_to_postgres, get_influx_data, connect_postgresql
from app.services.enrich_data import add_time_features, load_data_from_postgres, save_to_influx, query_influx
from app.services.lstm_model import MODEL, SCALER, SEQ_LENGTH, PREDICTION_DAYS, load_data_from_influx, load_model, save_predictions_to_db

app = FastAPI()

@app.get("/load-data")
async def load_data():
    data = load_energy_consumption_data()
    return {"nombre_de_lignes": len(data), "data": data.to_dict()}

@app.post("/forecast")
async def forecast(data: dict):
    df = load_data_from_influx()
    enriched_df = add_time_features(df)
    predictions = apply_arima_model(enriched_df)
    save_predictions_to_postgres(predictions)
    return {"forecast": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
