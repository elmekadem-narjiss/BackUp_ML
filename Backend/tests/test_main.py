import pytest
import pytest_asyncio
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_influx_data():
    mock_df = pd.DataFrame({
        "energyproduced": [100, 200, 300],
        "temperature": [20, 21, 22],
        "humidity": [50, 55, 60],
        "predicted_demand": [150, 250, 350],
        "demand": [140, 240, 340]
    })
    # Mocker le client InfluxDB pour éviter toute connexion réelle
    with patch("influxdb_client.InfluxDBClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.query_api.return_value.query_data_frame.return_value = mock_df
        # Mocker les fonctions spécifiques
        with patch("app.services.lstm_model.load_data_from_influx", return_value=mock_df):
            with patch("app.utils.time_series.load_energy_consumption_data", return_value=mock_df):
                yield mock_df

@pytest.mark.asyncio
async def test_load_data(mock_influx_data):
    try:
        response = client.get("/load-data")
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
        assert "energyproduced" in response.json(), f"Expected 'energyproduced' in response, got {response.json()}"
    except Exception as e:
        pytest.fail(f"Unexpected error in test_load_data: {str(e)}")

@pytest.mark.asyncio
async def test_forecast_data(mock_influx_data):
    try:
        response = client.post("/forecast", json={"energyproduced": 100, "temperature": 25, "humidity": 60})
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
        assert "prediction" in response.json(), f"Expected 'prediction' in response, got {response.json()}"
    except Exception as e:
        pytest.fail(f"Unexpected error in test_forecast_data: {str(e)}")

@pytest.mark.asyncio
async def test_forecast_no_data(mock_influx_data):
    try:
        response = client.post("/forecast", json={})
        assert response.status_code == 422, f"Expected status code 422, got {response.status_code}: {response.text}"
        assert "detail" in response.json(), f"Expected 'detail' in response, got {response.json()}"
    except Exception as e:
        pytest.fail(f"Unexpected error in test_forecast_no_data: {str(e)}")

@pytest.mark.asyncio
async def test_forecast_invalid_data(mock_influx_data):
    try:
        response = client.post("/forecast", json={"energyproduced": -100, "temperature": "invalid"})
        assert response.status_code == 422, f"Expected status code 422, got {response.status_code}: {response.text}"
        assert "detail" in response.json(), f"Expected 'detail' in response, got {response.json()}"
    except Exception as e:
        pytest.fail(f"Unexpected error in test_forecast_invalid_data: {str(e)}")
