import pytest
import numpy as np
import pandas as pd
import os
from train_ppo import train_ppo_model
from BESSBatteryEnv import BESSBatteryEnv
from stable_baselines3 import PPO

@pytest.fixture
def sample_csv(tmp_path):
    """Crée un fichier CSV temporaire pour les tests."""
    data = pd.DataFrame({
        'value': np.random.randn(100)
    })
    csv_path = tmp_path / "test_lstm_predictions.csv"
    data.to_csv(csv_path, index=False)
    return csv_path

def test_train_ppo_model(sample_csv, tmp_path):
    """Teste la fonction train_ppo_model."""
    output_dir = tmp_path / "output"
    model, metrics = train_ppo_model(str(sample_csv), str(output_dir), total_timesteps=100)
    
    # Vérifier que le modèle est une instance PPO
    assert isinstance(model, PPO)
    
    # Vérifier que les métriques sont générées
    assert isinstance(metrics, dict)
    assert "avg_reward" in metrics
    assert "avg_cycles" in metrics
    assert "avg_accuracy" in metrics
    
    # Vérifier que le fichier JSON des métriques existe
    metrics_path = output_dir / "ppo_bess_model_metrics.json"
    assert metrics_path.exists()
    
    # Vérifier que le modèle est sauvegardé
    model_path = output_dir / "ppo_model.zip"
    assert model_path.exists()
