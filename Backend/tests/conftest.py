import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_csv(tmp_path):
    """Fixture pour créer un fichier CSV de test avec les colonnes attendues."""
    df = pd.DataFrame({
        "energyproduced": np.random.uniform(100, 300, 100),
        "predicted_demand": np.random.uniform(150, 350, 100),
        "demand": np.random.uniform(140, 340, 100)
    })
    csv_path = tmp_path / "test_lstm_predictions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
