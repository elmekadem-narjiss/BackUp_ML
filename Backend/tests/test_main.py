import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import pandas as pd

client = TestClient(app)

@pytest.fixture
def fake_csv(tmp_path):
    # Créer un faux fichier CSV valide
    fake_data = """Year,Month,Day,Hour,Temperature,Humidity,SquareFootage,Occupancy,RenewableEnergy,EnergyConsumption,Timestamp
2024,4,6,12,22.5,45,100,1,5.2,10.5,2024-04-06 12:00:00
2024,4,6,13,23.0,44,100,0,5.5,11.0,2024-04-06 13:00:00
"""
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(fake_data)
    return file_path

@pytest.fixture(autouse=True)
def setup_file_csv(fake_csv, monkeypatch):
    # Redéfinir le chemin du fichier CSV dans app.main pour chaque test
    import app.main
    monkeypatch.setattr(app.main, "FILE_CSV", str(fake_csv))

def test_load_data(fake_csv):
    # Tester le chargement des données via l'API
    response = client.get("/load-data")
    
    # Vérifier le code de statut de la réponse
    assert response.status_code == 200
    
    # Vérifier que les données sont renvoyées correctement
    json_data = response.json()
    assert "nombre_de_lignes" in json_data
    assert "data" in json_data

    # Vérifier qu'on a bien les bonnes données dans la réponse
    assert len(json_data["data"]) > 0
    assert json_data["nombre_de_lignes"] == 2  # Comme il y a 2 lignes dans le CSV fictif

def test_forecast_data(fake_csv):
    # Charger les données
    client.get("/load-data")
    
    # Tester l'appel à /forecast
    response = client.get("/forecast")
    
    assert response.status_code == 200
    json_data = response.json()

    # Vérifier que les prévisions sont générées et retournées
    assert "message" in json_data
    assert json_data["message"] == "Prévisions générées et enregistrées avec succès."
    assert "forecast" in json_data
    assert isinstance(json_data["forecast"], dict)  # Vérifier que la prévision est un dictionnaire

def test_forecast_no_data(monkeypatch):
    # Mocker l'absence de données dans app.main
    import app.main
    monkeypatch.setattr(app.main, "data_cache", None)
    
    # Tester l'endpoint /forecast sans données chargées
    response = client.get("/forecast")
    assert response.status_code == 400
    assert response.json()["detail"] == "Colonnes manquantes : ['energyconsumption']"

def test_forecast_invalid_data(fake_csv):
    # Simuler un CSV sans la colonne 'energyconsumption'
    invalid_data = """Year,Month,Day,Hour,Temperature,Humidity,SquareFootage,Occupancy,RenewableEnergy
2024,4,6,12,22.5,45,100,1,5.2
2024,4,6,13,23.0,44,100,0,5.5
"""
    with open(fake_csv, "w") as f:
        f.write(invalid_data)

    # Charger les données invalides
    client.get("/load-data")

    # Tester l'appel à /forecast
    response = client.get("/forecast")
    assert response.status_code == 400
    assert "Colonnes manquantes" in response.json()["detail"]
