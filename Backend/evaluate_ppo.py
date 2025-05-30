import numpy as np
import json
from stable_baselines3 import PPO
from BESSBatteryEnv import BESSBatteryEnv
import os

def evaluate_ppo_model(model, file_path, output_dir, num_episodes=10):
    env = BESSBatteryEnv(file_path)
    total_rewards = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Observation non numérique ou invalide : {obs}")
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if not done and not np.all(np.isfinite(obs)):
                raise ValueError(f"Observation non numérique ou invalide après étape : {obs}")
            episode_reward += reward
        total_rewards.append(episode_reward)
    
    # Calculer les métriques
    metrics = {
        "total_reward": float(np.mean(total_rewards)),
        "cycles": float(num_episodes),
        "accuracy": 0.90  # Placeholder
    }
    
    # Sauvegarder les métriques
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    
    return metrics

if __name__ == "__main__":
    file_path = "lstm_predictions_charger.csv"
    output_dir = "output"
    model = PPO.load(os.path.join(output_dir, "ppo_model"))  # Charger le modèle entraîné
    metrics = evaluate_ppo_model(model, file_path, output_dir)
    print("Métriques d'évaluation :", metrics)
