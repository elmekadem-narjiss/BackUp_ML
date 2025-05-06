import pytest
import pandas as pd
import numpy as np
from app.services.lstm_model import prepare_data, build_model, train_and_save, load_model, save_predictions_to_db
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

@pytest.fixture
def sample_df():
    """Fixture pour un DataFrame de test."""
    return pd.DataFrame({
        "energyproduced": np.random.uniform(0, 1, 100),
        "temperature": np.random.uniform(0, 1, 100),
        "humidity": np.random.uniform(0, 1, 100),
        "month": np.random.randint(1, 13, 100),
        "week_of_year": np.random.randint(1, 53, 100),
        "hour": np.random.randint(0, 24, 100)
    })

def test_prepare_data(sample_df):
    """Teste la préparation des données."""
    X, y, scaler = prepare_data(sample_df)
    assert X.shape[0] > 0
    assert y.shape[0] > 0
    assert isinstance(scaler, MinMaxScaler)

def test_build_model():
    """Teste la construction du modèle."""
    model = build_model()
    assert isinstance(model, tf.keras.Sequential)
    assert len(model.layers) == 4  # LSTM, LSTM, Dense, Reshape

def test_train_and_save(sample_df, monkeypatch):
    """Teste l'entraînement et la sauvegarde du modèle."""
    monkeypatch.setattr("app.services.lstm_model.load_data_from_influx", lambda: sample_df)
    model, scaler, history, X, y = train_and_save()
    assert isinstance(model, tf.keras.Sequential)
    assert isinstance(scaler, MinMaxScaler)
    assert history is not None

def test_load_model(tmp_path, monkeypatch):
    """Teste le chargement du modèle et du scaler."""
    import pickle
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(60, 6))])
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
