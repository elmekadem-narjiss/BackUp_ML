import pytest
import pytest_asyncio
import respx
import httpx
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest_asyncio.fixture
async def mock_external_api():
    with respx.mock(base_url="https://api.example.com") as mock:
        mock.get("/data").mock(return_value=httpx.Response(200, json={"energyproduced": 100, "temperature": 25}))
        yield mock

@pytest.mark.asyncio
async def test_load_data(mock_external_api):
    response = client.get("/load-data")
    assert response.status_code == 200
    assert "energyproduced" in response.json()

@pytest.mark.asyncio
async def test_forecast_data(mock_external_api):
    response = client.post("/forecast", json={"energyproduced": 100, "temperature": 25, "humidity": 60})
    assert response.status_code == 200
    assert "prediction" in response.json()

@pytest.mark.asyncio
async def test_forecast_no_data(mock_external_api):
    response = client.post("/forecast", json={})
    assert response.status_code == 400
    assert "error" in response.json()

@pytest.mark.asyncio
async def test_forecast_invalid_data(mock_external_api):
    response = client.post("/forecast", json={"energyproduced": -100, "temperature": "invalid"})
    assert response.status_code == 422
    assert "detail" in response.json()
