import pytest
import pytest_asyncio
import respx
import httpx
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest_asyncio.fixture
async def mock_external_api():
    with respx.mock(base_url="https://api.example.com") as mock:
        mock.get("/data").mock(return_value=httpx.Response(200, json={"energyproduced": 100, "temperature": 25}))
        yield mock

@pytest.fixture
def mock_influx_data(mocker):
    mock_df = pd.DataFrame({
        "energyproduced": [100, 200, 300],
        "temperature": [20, 21, 22],
        "humidity": [50, 55, 60]
    })
    mocker.patch("app.services.lstm_model.load_data_from_influx", return_value=mock_df)
    return mock_df

@pytest.mark.asyncio
async def test_load_data(mock_external_api, mock_influx_data):
    response = client.get("/load-data")
    assert response.status_code == 200
    assert "energyproduced" in response.json()

@pytest.mark.asyncio
async def test_forecast_data(mock_external_api, mock_influx_data):
    response = client.post("/forecast", json={"energyproduced": 100, "temperature": 25, "humidity": 60})
    assert response.status_code == 200
    assert "prediction" in response.json()

@pytest.mark.asyncio
async def test_forecast_no_data(mock_external_api, mock_influx_data):
    response = client.post("/forecast", json={})
    assert response.status_code == 400
    assert "error" in response.json()

@pytest.mark.asyncio
async def test_forecast_invalid_data(mock_external_api, mock_influx_data):
    response = client.post("/forecast", json={"energyproduced": -100, "temperature": "invalid"})
    assert response.status_code == 422
    assert "detail" in response.json()
