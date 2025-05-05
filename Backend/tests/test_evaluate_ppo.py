import pytest
   import numpy as np
   import pandas as pd
   import os
   from evaluate_ppo import evaluate_ppo_model
   from train_ppo import train_ppo_model
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

   @pytest.fixture
   def trained_model(sample_csv, tmp_path):
       """Entraîne un modèle PPO pour les tests."""
       output_dir = tmp_path / "output"
       model, _ = train_ppo_model(str(sample_csv), str(output_dir), total_timesteps=100)
       return model, str(output_dir)

   def test_evaluate_ppo_model(trained_model, sample_csv, tmp_path):
       """Teste la fonction evaluate_ppo_model."""
       model, output_dir = trained_model
       metrics = evaluate_ppo_model(model, str(sample_csv), output_dir, num_episodes=5)
       
       # Vérifier que les métriques sont générées
       assert isinstance(metrics, dict)
       assert "total_reward" in metrics
       assert "cycles" in metrics
       assert "accuracy" in metrics
       assert metrics["cycles"] == 5.0
       
       # Vérifier que le fichier JSON des métriques existe
       metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
       assert os.path.exists(metrics_path)
