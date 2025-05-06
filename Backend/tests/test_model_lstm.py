import pytest
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from unittest.mock import MagicMock, patch
import pickle
import pandas as pd
from app.services.lstm_model import prepare_data, build_model, train_and_save, load_model, save_predictions_to_db

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "energyproduced": [100, 200, 300, 400, 500],
        "temperature": [20, 22, 24, 23, 25],
        "humidity": [50, 55, 60, 58, 62],
        "month": [1, 1, 1, 2, 2],
        "week_of_year": [1, 1, 2, 2, 3],
        "hour": [0, 6, 12, 18, 0]
    })

def test_prepare_data(sample_df):
    """Teste la préparation des données pour l'entraînement."""
    X, y, scaler = prepare_data(sample_df)
    assert X.shape[0] == len(sample_df) - 60
    assert y.shape[0] == len(sample_df) - 60
    assert isinstance(scaler, MinMaxScaler)

def test_build_model():
    """Teste la construction du modèle LSTM."""
    model = build_model()
    assert isinstance(model, tf.keras.Sequential)
    assert len(model.layers) == 4

def test_train_and_save(tmp_path, sample_df):
    """Teste l'entraînement et la sauvegarde du modèle."""
    model_path = tmp_path / "model.keras"
    scaler_path = tmp_path / "scaler.pkl"
    model, scaler = train_and_save(sample_df, str(model_path), str(scaler_path), epochs=1)
    assert isinstance(model, tf.keras.Sequential)
    assert isinstance(scaler, MinMaxScaler)
    assert model_path.exists()
    assert scaler_path.exists()

def test_load_model(tmp_path, monkeypatch):
    """Teste le chargement du modèle et du scaler."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 6)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.save(tmp_path / "model.keras")
    with open(tmp_path / "scaler.pkl", "wb") as f:
        pickle.dump(MinMaxScaler(), f)
    monkeypatch.setattr("app.services.lstm_model.MODEL_PATH", str(tmp_path / "model.keras"))
    monkeypatch.setattr("app.services.lstm_model.SCALER_PATH", str(tmp_path / "scaler.pkl"))
    loaded_model, loaded_scaler = load_model()
    assert isinstance(loaded_model, tf.keras.Sequential)
    assert isinstance(loaded_scaler, MinMaxScaler)

def test_save_predictions_to_db(monkeypatch):
    """Teste la sauvegarde des prédictions dans PostgreSQL."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    monkeypatch.setattr("app.services.lstm_model.connect_postgresql", lambda: mock_conn)
    predictions = [
        {
            "energyproduced": 100.0,
            "temperature": 20.0,
            "humidity": 50.0,
            "month": 1,
            "week_of_year": 1,
            "hour": 12
        }
    ]
    save_predictions_to_db(predictions)
    mock_cursor.execute.assert_called()
    mock_conn.commit.assert_called()
