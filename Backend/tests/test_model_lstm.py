import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from app.services.lstm_model import (
    build_model,
    prepare_data,
    train_and_save,
    load_model,
    save_predictions_to_db,
)

# Fixture pour créer un DataFrame de test
@pytest.fixture
def sample_df():
    data = {
        "energyproduced": np.random.rand(100),
        "temperature": np.random.rand(100),
        "humidity": np.random.rand(100),
        "month": np.random.randint(1, 13, 100),
        "week_of_year": np.random.randint(1, 53, 100),
        "hour": np.random.randint(0, 24, 100),
    }
    return pd.DataFrame(data)

# Test de la fonction prepare_data
def test_prepare_data(sample_df):
    X, y, scaler = prepare_data(sample_df)
    assert X.shape[1:] == (60, 6)
    assert y.shape[1:] == (30, 6)
    assert X.shape[0] == y.shape[0]

# Test de build_model
def test_build_model():
    model = build_model()
    assert model is not None
    assert model.input_shape == (None, 60, 6)

# Test de train_and_save avec monkeypatch
def test_train_and_save(sample_df, monkeypatch):
    monkeypatch.setattr("app.services.lstm_model.load_data_from_influx", lambda: sample_df)
    model, scaler = train_and_save()
    assert model is not None
    assert scaler is not None

# Test de load_model avec modèle fictif
def test_load_model(tmp_path, monkeypatch):
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    import pickle
    # Créer un modèle fictif
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(60, 6))])
    model.save(tmp_path / "model")
    # Créer un scaler fictif
    scaler = MinMaxScaler()
    with open(tmp_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    # Mocker les chemins de chargement
    monkeypatch.setattr("app.services.lstm_model.MODEL_PATH", str(tmp_path / "model"))
    monkeypatch.setattr("app.services.lstm_model.SCALER_PATH", str(tmp_path / "scaler.pkl"))
    loaded_model, loaded_scaler = load_model()
    assert loaded_model is not None
    assert loaded_scaler is not None

# Test de save_predictions_to_db avec patch de la DB
def test_save_predictions_to_db(monkeypatch):
    fake_conn = MagicMock()
    fake_cursor = MagicMock()
    fake_conn.cursor.return_value = fake_cursor
    monkeypatch.setattr("app.services.lstm_model.connect_postgresql", lambda: fake_conn)

    predictions = [{
        "energyproduced": 0.5,
        "temperature": 0.3,
        "humidity": 0.2,
        "month": 5,
        "week_of_year": 20,
        "hour": 14
    } for _ in range(5)]

    save_predictions_to_db(predictions)
    assert fake_cursor.execute.called
