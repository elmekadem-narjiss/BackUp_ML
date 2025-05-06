import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch
import logging

from app.main import app

logger = logging.getLogger(__name__)
client = TestClient(app)

@pytest.fixture
def mock_influx_data():
    mock_df = pd.DataFrame({
        'energyproduced': [100, 200, 300],
        'temperature': [20, 22, 24],
        'humidity': [50, 55, 60],
        'predicted_demand': [250, 350, 450],
        'demand': [240, 340, 440]
    }, index=pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00']))
    with patch("app.services.lstm_model.InfluxDBClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        mock_query_api = mock_instance.query_api.return_value
        mock_query_api.query_data_frame.return_value = mock_df
        with patch("app.utils.time_series.load_energy_consumption_data", return_value=mock_df):
            with patch("app.services.lstm_model.load_data_from_influx", return_value=mock_df):
                yield mock_df

@pytest.mark.asyncio
async def test_load_data(mock_influx_data):
    logger.debug("Running test_load_data")
    response = client.get("/load-data")
    logger.debug(f"Response from /load-data: status={response.status_code}, json={response.json()}")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert response.json()["energyproduced"] == [100, 200, 300]

@pytest.mark.asyncio
async def test_forecast_data(mock_influx_data):
    logger.debug("Running test_forecast_data")
    response = client.post("/forecast", json={"energyproduced": 100, "temperature": 25, "humidity": 60})
    logger.debug(f"Response from /forecast: status={response.status_code}, json={response.json()}")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "prediction" in response.json()

@pytest.mark.asyncio
async def test_forecast_no_data(mock_influx_data):
    logger.debug("Running test_forecast_no_data")
    response = client.post("/forecast", json={})
    logger.debug(f"Response from /forecast (no data): status={response.status_code}, json={response.json()}")
    assert response.status_code == 422
    assert isinstance(response.json(), dict)
    assert "detail" in response.json()

@pytest.mark.asyncio
async def test_forecast_invalid_data(mock_influx_data):
    logger.debug("Running test_forecast_invalid_data")
    response = client.post("/forecast", json={"energyproduced": -100, "temperature": "invalid"})
    logger.debug(f"Response from /forecast (invalid data): status={response.status_code}, json={response.json()}")
    assert response.status_code == 422
    assert isinstance(response.json(), dict)
    assert "detail" in response.json()
