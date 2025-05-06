import pytest
import pandas as pd
from unittest.mock import patch
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
    }, index=pd.date_range(start="2023-01-01", periods=3, freq="h"))
    
    with patch("influxdb_client.InfluxDBClient") as mock_client:
        mock_instance = mock_client.return_value
        mock_query_api = mock_instance.query_api.return_value
        mock_query_api.query_data_frame.return_value = mock_df
        with patch("app.utils.time_series.load_energy_consumption_data", return_value=mock_df):
            with patch("app.services.lstm_model.load_data_from_influx", return_value=mock_df):
                yield mock_df

@pytest.mark.asyncio
async def test_load_data(mock_influx_data):
    response = client.get("/load-data")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "energyproduced" in response.json()

@pytest.mark.asyncio
async def test_forecast_data(mock_influx_data):
    response = client.post("/forecast", json={"energyproduced": 100, "temperature": 25, "humidity": 60})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "prediction" in response.json()

@pytest.mark.asyncio
async def test_forecast_no_data(mock_influx_data):
    response = client.post("/forecast", json={})
    assert response.status_code == 422
    assert isinstance(response.json(), dict)
    assert "detail" in response.json()

@pytest.mark.asyncio
async def test_forecast_invalid_data(mock_influx_data):
    response = client.post("/forecast", json={"energyproduced": -100, "temperature": "invalid"})
    assert response.status_code == 422
    assert isinstance(response.json(), dict)
    assert "detail" in response.json()
