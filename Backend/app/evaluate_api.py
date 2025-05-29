from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import io
import base64
import os
import gdown

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_file_from_drive(file_id: str, output_path: str) -> None:
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        logging.info(f"Fichier téléchargé avec succès sous {output_path}")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Le fichier {output_path} n'a pas été téléchargé correctement.")
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement : {e}")
        raise Exception(f"Erreur de téléchargement : {e}")

def fetch_lstm_predictions(file_path: str):
    try:
        logging.debug("Chargement des données...")
        df = pd.read_csv(file_path)
        logging.info("Données chargées avec succès.")
        df = df.dropna()
        df["future_production"] = df["energyproduced"].shift(-3).fillna(method='ffill')
        columns_to_normalize = ["energyproduced", "temperature", "humidity", "month", "week_of_year", "hour", "future_production"]
        for col in columns_to_normalize:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df[col] = 100 * (df[col] - col_min) / (col_max - col_min)
            else:
                df[col] = 0
        return df
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {e}")
        raise Exception(f"Erreur de chargement : {e}")

class BESSBatteryEnv(gym.Env):
    def __init__(self, data, battery_capacity=100, charge_efficiency=0.95, discharge_efficiency=0.95, soc_min=20, soc_max=80):
        super(BESSBatteryEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.index = 0
        self.max_index = len(data) - 1
        self.battery_capacity = battery_capacity
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc = 50
        self.cycle_count = 0
        self.max_cycles = 3000
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.optimal_actions = 0

    def reset(self):
        self.index = 0
        self.soc = 50
        self.cycle_count = 0
        self.previous_action = None
        self.optimal_actions = 0
        return self._get_observation()

    def _get_observation(self):
        row = self.data.iloc[self.index]
        return np.array([
            row["energyproduced"],
            row["temperature"],
            row["humidity"],
            row["month"],
            row["week_of_year"],
            row["hour"],
            self.soc,
            row["future_production"]
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        charge_rate = 5
        current = self.data.iloc[self.index]
        future = self.data.iloc[min(self.index + 3, self.max_index)]
        production_diff = abs(future["future_production"] - current["energyproduced"]) / 100
        optimal_action = 0
        if production_diff > 0.05:
            optimal_action = 1 if future["future_production"] > current["energyproduced"] else 2
        if 17 <= current["hour"] <= 20 and self.soc > 20:
            optimal_action = 2
        if self.soc < 20:
            optimal_action = 1
        elif self.soc > 80:
            optimal_action = 2
        if action == optimal_action:
            self.optimal_actions += 1
        if action == 1 and self.soc < 100:
            charge_amount = min(charge_rate, 100 - self.soc) * self.charge_efficiency
            self.soc += charge_amount
            production_diff = (future["future_production"] - current["energyproduced"]) / 100
            reward = 5 * production_diff if future["future_production"] > current["energyproduced"] else -1
        elif action == 2 and self.soc > 0:
            discharge_amount = min(charge_rate, self.soc) * self.discharge_efficiency
            self.soc -= discharge_amount
            production_diff = (current["energyproduced"] - future["future_production"]) / 100
            reward = 5 * production_diff if future["future_production"] < current["energyproduced"] else -1
        else:
            production_diff = abs(future["future_production"] - current["energyproduced"]) / 100
            if production_diff > 0.05:
                reward = -1.0
            else:
                reward = 0
        if self.previous_action is not None and action != self.previous_action and action != 0:
            reward -= 0.002
            if (self.previous_action == 1 and action == 2) or (self.previous_action == 2 and action == 1):
                self.cycle_count += 0.5
        self.previous_action = action
        if self.soc < 25 or self.soc > 75:
            reward -= 1.0
        if self.soc < self.soc_min or self.soc > self.soc_max:
            reward -= 2.0
        elif self.soc_min + 10 <= self.soc <= self.soc_max - 10:
            reward += 0.75
        if action == 2 and 17 <= current["hour"] <= 20:
            reward += 1.5
        if (action == 1 or action == 2) and current["temperature"] > 75:
            reward -= 0.5
        if self.cycle_count > self.max_cycles * 0.8:
            reward -= 2
        self.index += 1
        if self.index >= self.max_index:
            done = True
            self.accuracy = (self.optimal_actions / 85) * 100
        return self._get_observation(), reward, done, {}

def load_and_evaluate_model(file_path: str, model_path: str):
    try:
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"Modèle {model_path}.zip non trouvé.")
        df = fetch_lstm_predictions(file_path)
        env = DummyVecEnv([lambda: BESSBatteryEnv(df)])
        model = PPO.load(model_path)
        logging.info("Modèle chargé avec succès.")
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
            socs.append(float(obs[0][6]))
            actions_taken.append(int(action.item()))
            future_productions.append(float(obs[0][7]))
            if previous_action is not None:
                if (previous_action == 1 and action == 2) or (previous_action == 2 and action == 1):
                    cycle_count += 0.5
            previous_action = action
            results.append({
                "Step": len(results) + 1,
                "Action": ["Rester", "Charger", "Décharger"][int(action)],
                "SOC (%)": float(obs[0][6]),
                "Future Production (%)": float(obs[0][7]),
                "Reward": float(reward)
            })
        total_reward = float(total_reward)
        cycle_count = float(cycle_count)
        accuracy = float(env.envs[0].accuracy)
        fig = plt.figure(figsize=(14, 12))
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
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_data = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return {
            "metrics": {
                "total_reward": total_reward,
                "cycle_count": cycle_count,
                "accuracy": accuracy,
                "soc_final": socs[-1]
            },
            "results": results,
            "graph_data": graph_data
        }
    except FileNotFoundError as e:
        logging.error(f"Erreur : {e}")
        raise Exception(str(e))
    except Exception as e:
        logging.error(f"Erreur lors de l'évaluation : {e}")
        raise Exception(f"Erreur d'évaluation : {e}")

@app.get("/evaluate")
async def evaluate():
    try:
        model_file_id = "1-A68xVarqwE0Sw1H6zEXV9zBYhG1xxHK"
        data_file_id = "1nT6AH5scHrteA7LkumeSaylsHpnX-X1d"
        model_path = "D:/PFE/Backend_ML/ppo_bess_model"
        data_path = "D:/PFE/Backend_ML/lstm_predictions_charger.csv"
        if not os.path.exists(model_path + ".zip"):
            download_file_from_drive(model_file_id, model_path + ".zip")
        if not os.path.exists(data_path):
            download_file_from_drive(data_file_id, data_path)
        result = load_and_evaluate_model(data_path, model_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)