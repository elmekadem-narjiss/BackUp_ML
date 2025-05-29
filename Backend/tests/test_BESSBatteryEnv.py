import pytest
import numpy as np
import pandas as pd
from BESSBatteryEnv import BESSBatteryEnv

@pytest.fixture
def sample_csv(tmp_path):
    """Crée un fichier CSV temporaire pour les tests."""
    data = pd.DataFrame({
        'energyproduced': np.random.randn(100) * 100  # Realistic values for energy
    })
    csv_path = tmp_path / "test_lstm_predictions.csv"
    data.to_csv(csv_path, index=False)
    return csv_path

def test_bess_battery_env(sample_csv):
    """Teste l'initialisation et les étapes de BESSBatteryEnv."""
    env = BESSBatteryEnv(str(sample_csv))
    
    # Vérifier l'initialisation
    assert env.capacity == 100.0
    assert env.soc == 50.0
    assert env.current_step == 0
    assert env.max_steps == 99  # len(data) - 1
    
    # Vérifier l'espace d'action
    assert env.action_space.shape == (1,)
    assert env.action_space.low == pytest.approx(-1.0)
    assert env.action_space.high == pytest.approx(1.0)
    
    # Vérifier l'espace d'observation
    assert env.observation_space.shape == (1,)  # Une colonne 'energyproduced'
    
    # Tester une étape
    obs, reward, done, _ = env.step([0.5])
    assert not done
    assert isinstance(obs, np.ndarray)
    assert np.all(np.isfinite(obs))
    assert isinstance(reward, float)
    
    # Tester la récompense
    reward = env._calculate_reward([0.5])
    assert isinstance(reward, float)
    assert reward != -10.0  # Pas de pénalité pour une action valide
