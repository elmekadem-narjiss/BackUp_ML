import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.lstm_model import MODEL, SCALER, SEQ_LENGTH, PREDICTION_DAYS, load_model, save_predictions_to_db

client = TestClient(app)

@pytest.mark.asyncio
async def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the LSTM Prediction API"}

@pytest.mark.asyncio
async def test_predict_endpoint(mocker):
    # Mock load_model and save_predictions_to_db
    mocker.patch("app.services.lstm_model.load_model", return_value=MODEL)
    mocker.patch("app.services.lstm_model.save_predictions_to_db", return_value=None)
    
    # Sample input data
    input_data = {
        "data": [[100, 200, 300]] * SEQ_LENGTH,
        "days": PREDICTION_DAYS
    }
    
    response = client.post("/predict/", json=input_data)
    assert response.status_code == 200
    assert "predictions" in response.json()
