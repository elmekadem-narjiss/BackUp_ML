import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import MagicMock

client = TestClient(app)

@pytest.fixture
def fake_csv(tmp_path):
    import pandas as pd
    data = {
        "timestamp": ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        "energyproduced": [100.0, 120.0],
        "temperature": [20.0, 21.0],
        "humidity": [50.0, 55.0],
        "month": [1, 1],
        "week_of_year": [1, 1],
        "hour": [0, 1]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_load_data(fake_csv, monkeypatch):
    # Mocker les connexions à InfluxDB, PostgreSQL, Redis, et MQTT
    monkeypatch.setattr("app.database.db_connection.connect_influxdb", lambda: MagicMock())
    monkeypatch.setattr("app.database.db_connection.connect_postgresql", lambda: MagicMock())
    monkeypatch.setattr("app.database.db_connection.redis.Redis", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("app.database.db_connection.connect_mqtt", lambda: MagicMock())
    response = client.get("/load-data")
    assert response.status_code == 200
    json_data = response.json()
    assert "nombre_de_lignes" in json_data
    assert "data" in json_data
    assert len(json_data["data"]) > 0
    assert json_data["nombre_de_lignes"] == 2

def test_forecast_data(fake_csv, monkeypatch):
    # Mocker les connexions
    monkeypatch.setattr("app.database.db_connection.connect_influxdb", lambda: MagicMock())
    monkeypatch.setattr("app.database.db_connection.connect_postgresql", lambda: MagicMock())
    monkeypatch.setattr("app.database.db_connection.redis.Redis", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("app.database.db_connection.connect_mqtt", lambda: MagicMock())
    response = client.post("/forecast", json={"data": "test_data"})
    assert response.status_code == 200
    json_data = response.json()
    assert "forecast" in json_data

def test_forecast_no_data(monkeypatch):
    # Mocker les connexions
    monkeypatch.setattr("app.database.db_connection.connect_influxdb", lambda: MagicMock())
    monkeypatch.setattr("app.database.db_connection.connect_postgresql", lambda: MagicMock())
    monkeypatch.setattr("app.database.db_connection.redis.Redis", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("app.database.db_connection.connect_mqtt", lambda: MagicMock())
    response = client.post("/forecast", json={})
    assert response.status_code == 422

def test_forecast_invalid_data(monkeypatch):
    # Mocker les connexions
    monkeypatch.setattr("app.database.db_connection.connect_influxdb", lambda: MagicMock())
    monkeypatch.setattr("app.database.db_connection.connect_postgresql", lambda: MagicMock())
    monkeypatch.setattr("app.database.db_connection.redis.Redis", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("app.database.db_connection.connect_mqtt", lambda: MagicMock())
    response = client.post("/forecast", json={"data": 123})
    assert response.status_code == 422
