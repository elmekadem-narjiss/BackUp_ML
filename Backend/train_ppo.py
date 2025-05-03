import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os
import json
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from BESSBatteryEnv import BESSBatteryEnv, fetch_lstm_predictions_from_drive

def train_ppo_model(file_path, mlflow_url, output_dir, model_path):
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("PPO_BESS_Training")
    
    with mlflow.start_run():
        df = fetch_lstm_predictions_from_drive(file_path)
        env = DummyVecEnv([lambda: BESSBatteryEnv(df)])
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.00005,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
        model.learn(total_timesteps=3500000)
        model.save(model_path)
        print(f"Entraînement terminé. Modèle sauvegardé dans {model_path}")

        rewards, socs, actions_taken, cycle_counts, future_productions, accuracies = visualize_training(model, env)
        
        # Logger les métriques dans MLflow
        mlflow.log_param("learning_rate", 0.00005)
        mlflow.log_param("total_timesteps", 3500000)
        mlflow.log_metric("avg_reward", np.mean(rewards))
        mlflow.log_metric("avg_cycles", np.mean(cycle_counts))
        mlflow.log_metric("avg_accuracy", np.mean(accuracies))
        
        # Sauvegarder les graphiques dans output_dir et les logger dans MLflow
        plots_path = os.path.join(output_dir, "training_plots.png")
        plt.savefig(plots_path)
        mlflow.log_artifact(plots_path)
        
        # Sauvegarder les métriques dans un fichier JSON
        metrics = {
            "avg_reward": float(np.mean(rewards)),
            "avg_cycles": float(np.mean(cycle_counts)),
            "avg_accuracy": float(np.mean(accuracies))
        }
        metrics_path = os.path.join(output_dir, "ppo_bess_model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        return metrics

def visualize_training(model, env, num_episodes=3):
    rewards = []
    socs = []
    actions_taken = []
    cycle_counts = []
    future_productions = []
    accuracies = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        episode_socs = []
        episode_actions = []
        episode_future_prods = []
        cycle_count = 0
        previous_action = None

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            episode_socs.append(obs[0][6])
            episode_actions.append(int(action.item()))
            episode_future_prods.append(obs[0][7])

            if previous_action is not None:
                if (previous_action == 1 and action == 2) or (previous_action == 2 and action == 1):
                    cycle_count += 0.5
            previous_action = action

        rewards.append(total_reward)
        socs.append(episode_socs)
        actions_taken.append(episode_actions)
        cycle_counts.append(cycle_count)
        future_productions.append(episode_future_prods)
        accuracies.append(env.envs[0].accuracy)
        print(f"Épisode {episode+1} ➤ Récompense totale : {total_reward}, Cycles : {cycle_count}, Accuracy : {env.envs[0].accuracy:.2f}%")

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_cycles = np.mean(cycle_counts)
    avg_accuracy = np.mean(accuracies)
    print(f"Moyenne des récompenses : {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Moyenne des cycles par épisode : {avg_cycles:.2f}")
    print(f"Moyenne de l'accuracy : {avg_accuracy:.2f}%")

    plt.figure(figsize=(14, 10))
    plt.subplot(3, 1, 1)
    for i, soc in enumerate(socs):
        plt.plot(soc, label=f"Épisode {i+1}")
    plt.axhline(y=20, color='r', linestyle='--', label='SOC min')
    plt.axhline(y=80, color='r', linestyle='--', label='SOC max')
    plt.title("Évolution du SOC")
    plt.xlabel("Temps (étapes)")
    plt.ylabel("SOC (%)")
    plt.legend()

    plt.subplot(3, 1, 2)
    for i, actions in enumerate(actions_taken):
        plt.plot(actions, label=f"Épisode {i+1}")
    plt.title("Actions prises (0 = Rester, 1 = Charger, 2 = Décharger)")
    plt.xlabel("Temps (étapes)")
    plt.ylabel("Action")
    plt.yticks([0, 1, 2], ['Rester', 'Charger', 'Décharger'])
    plt.legend()

    plt.subplot(3, 1, 3)
    for i, prod in enumerate(future_productions):
        plt.plot(prod, label=f"Épisode {i+1}")
    plt.title("Production énergétique future normalisée")
    plt.xlabel("Temps (étapes)")
    plt.ylabel("Production future (%)")
    plt.legend()

    plt.tight_layout()
    return rewards, socs, actions_taken, cycle_counts, future_productions, accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Chemin vers le fichier de prédictions LSTM")
    parser.add_argument("--mlflow_url", type=str, required=True, help="URL du serveur MLflow")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier de sortie")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin où sauvegarder le modèle")
    args = parser.parse_args()

    train_ppo_model(args.file_path, args.mlflow_url, args.output_dir, args.model_path)