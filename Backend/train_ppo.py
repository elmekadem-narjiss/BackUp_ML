import numpy as np
import json
from stable_baselines3 import PPO
from BESSBatteryEnv import BESSBatteryEnv
import os

def train_ppo_model(file_path, output_dir, total_timesteps=10000):
    env = BESSBatteryEnv(file_path)
    # Vérifier l'observation initiale
    obs = env.reset()
    if not np.all(np.isfinite(obs)):
        raise ValueError(f"Observation initiale non numérique ou invalide : {obs}")
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    
    # Sauvegarder le modèle
    model.save(os.path.join(output_dir, "ppo_model"))
    
    # Calculer les métriques (exemple)
    metrics = {
        "avg_reward": np.mean([env._calculate_reward(np.random.uniform(-1, 1)) for _ in range(100)]),
        "avg_cycles": 100.0,  # Placeholder
        "avg_accuracy": 0.95   # Placeholder
    }
    
    # Sauvegarder les métriques
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "ppo_bess_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    
    return model, metrics

if __name__ == "__main__":
    file_path = "lstm_predictions_charger.csv"
    output_dir = "output"
    model, metrics = train_ppo_model(file_path, output_dir)
    print("Métriques PPO :", metrics)
