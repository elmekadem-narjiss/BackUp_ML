import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock
import httpx
import respx

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_influxdb():
    with patch("app.database.db_connection.connect_influxdb") as mock:
        mock_client = MagicMock()
        mock_client.query_api().query_data_frame.return_value = pd.DataFrame({
            "timestamp": ["2023-01-01"],
            "energyproduced": [100.0],
            "temperature": [20.0],
            "humidity": [50.0],
            "month": [1],
            "week_of_year": [1],
            "hour": [12]
        })
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def mock_postgres():
    with patch("app.database.db_connection.connect_postgresql") as mock:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock.return_value = mock_conn
        yield mock

@pytest.fixture
def mock_redis():
    with patch("app.database.db_connection.connect_redis") as mock:
        mock_redis = MagicMock()
        mock.return_value = mock_redis
        yield mock

@pytest.fixture
def mock_mqtt():
    with patch("app.database.db_connection.connect_mqtt") as mock:
        mock_mqtt = MagicMock()
        mock.return_value = mock_mqtt
        yield mock

@pytest_asyncio.fixture
async def mock_http():
    with respx.mock(base_url="http://localhost:8000") as respx_mock:
        respx_mock.get("/load-data").respond(
            status_code=200,
            json={"nombre_de_lignes": 1, "data": {"timestamp": ["2023-01-01"], "energyproduced": [100.0]}}
        )
        respx_mock.post("/forecast").respond(
            status_code=200,
            json={"forecast": {"energyproduced": [110.0]}}
        )
        yield respx_mock

@pytest.mark.asyncio
async def test_load_data(client, mock_influxdb, mock_postgres, mock_redis, mock_mqtt, mock_http):
    async with httpx.AsyncClient(app=app, base_url="http://localhost:8000") as async_client:
        response = await async_client.get("/load-data")
        assert response.status_code == 200
        assert response.json()["nombre_de_lignes"] > 0

@pytest.mark.asyncio
async def test_forecast_data(client, mock_influxdb, mock_postgres, mock_redis, mock_mqtt, mock_http):
    async with httpx.AsyncClient(app=app, base_url="http://localhost:8000") as async_client:
        response = await async_client.post("/forecast", json={"data": {"timestamp": "2023-01-01", "energyproduced": 100.0}})
        assert response.status_code == 200
        assert "forecast" in response.json()

@pytest.mark.asyncio
async def test_forecast_no_data(client, mock_influxdb, mock_postgres, mock_redis, mock_mqtt, mock_http):
    async with httpx.AsyncClient(app=app, base_url="http://localhost:8000") as async_client:
        response = await async_client.post("/forecast", json={})
        assert response.status_code == 200
        assert response.json().get("forecast") is None

@pytest.mark.asyncio
async def test_forecast_invalid_data(client, mock_influxdb, mock_postgres, mock_redis, mock_mqtt, mock_http):
    async with httpx.AsyncClient(app=app, base_url="http://localhost:8000") as async_client:
        response = await async_client.post("/forecast", json={"data": {"invalid": "data"}})
        assert response.status_code == 200
        assert response.json().get("forecast") is None
