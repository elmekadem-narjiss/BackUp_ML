import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from BESSBatteryEnv import BESSBatteryEnv, fetch_lstm_predictions_from_drive

logging.basicConfig(level=logging.INFO)

def load_and_evaluate_model(file_path="/content/drive/MyDrive/lstm_predictions_charger.csv", model_path="/content/drive/MyDrive/ppo_bess_model", mlflow_url="http://localhost:5000"):
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("PPO_BESS_Evaluation")
    
    with mlflow.start_run():
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"Modèle {model_path} non trouvé.")

        df = fetch_lstm_predictions_from_drive(file_path)
        env = DummyVecEnv([lambda: BESSBatteryEnv(df)])

        model = PPO.load(model_path)
        print("Modèle chargé avec succès.")

        obs = env.reset()
        done = False
        total_reward = 0
        socs = []
        actions_taken = []
        future_productions = []
        cycle_count = 0
        previous_action = None
        results = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            socs.append(obs[0][6])
            actions_taken.append(int(action.item()))
            future_productions.append(obs[0][7])

            if previous_action is not None:
                if (previous_action == 1 and action == 2) or (previous_action == 2 and action == 1):
                    cycle_count += 0.5
            previous_action = action

            results.append({
                "Step": len(results) + 1,
                "Action": ["Rester", "Charger", "Décharger"][int(action)],
                "SOC (%)": obs[0][6],
                "Future Production (%)": obs[0][7],
                "Reward": reward
            })

        accuracy = env.envs[0].accuracy
        print(f"Évaluation terminée ➤ Récompense totale : {total_reward:.2f}, Cycles : {cycle_count:.1f}, Accuracy : {accuracy:.2f}%")
        print(f"SOC final : {socs[-1]:.2f}%")

        results_df = pd.DataFrame(results)
        print("\nTableau des résultats :")
        print(results_df.to_string(index=False))

        plt.figure(figsize=(14, 12))
        plt.subplot(3, 1, 1)
        plt.plot(socs, label="SOC")
        plt.axhline(y=20, color='r', linestyle='--', label='SOC min')
        plt.axhline(y=80, color='r', linestyle='--', label='SOC max')
        plt.title("Évolution du SOC")
        plt.xlabel("Temps (étapes)")
        plt.ylabel("SOC (%)")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(actions_taken, label="Actions")
        plt.title("Actions prises (0 = Rester, 1 = Charger, 2 = Décharger)")
        plt.xlabel("Temps (étapes)")
        plt.ylabel("Action")
        plt.yticks([0, 1, 2], ['Rester', 'Charger', 'Décharger'])
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(future_productions, label="Production future")
        plt.title("Production énergétique future normalisée")
        plt.xlabel("Temps (étapes)")
        plt.ylabel("Production future (%)")
        plt.legend()

        plt.tight_layout()
        plt.savefig("/content/drive/MyDrive/evaluation_plots.png")
        mlflow.log_artifact("/content/drive/MyDrive/evaluation_plots.png")

        mlflow.log_metric("total_reward", total_reward)
        mlflow.log_metric("cycles", cycle_count)
        mlflow.log_metric("accuracy", accuracy)

        # Sauvegarder les métriques dans un fichier JSON pour Snakemake
        metrics = {
            "total_reward": float(total_reward),
            "cycles": float(cycle_count),
            "accuracy": float(accuracy)
        }
        import json
        with open("/content/drive/MyDrive/evaluation_metrics.json", "w") as f:
            json.dump(metrics, f)

        return metrics

if __name__ == "__main__":
    load_and_evaluate_model()