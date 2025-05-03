import gym
import numpy as np
import pandas as pd
import logging
from google.colab import drive

logging.basicConfig(level=logging.INFO)

def mount_google_drive():
    try:
        drive.mount('/content/drive')
        logging.info("Google Drive monté avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors du montage de Google Drive : {e}")
        raise Exception(f"Erreur Google Drive : {e}")

def fetch_lstm_predictions_from_drive(file_path="/content/drive/MyDrive/lstm_predictions_charger.csv"):
    try:
        logging.debug("Chargement des données depuis Google Drive...")
        df = pd.read_csv(file_path)
        logging.info("Données chargées avec succès.")

        if len(df) < 85:
            raise ValueError(f"Le fichier doit contenir au moins 85 lignes, trouvé : {len(df)} lignes")

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
        print(f"Step {self.index}: production_diff = {production_diff:.3f}, Current SOC = {self.soc:.2f}, Action = {action}")

        optimal_action = 0
        if production_diff > 0.05:
            optimal_action = 1 if future["future_production"] > current["energyproduced"] else 2
        if future["future_production"] > 70 and self.soc < 80:
            optimal_action = 1
        if 16 <= current["hour"] <= 21 and self.soc > 20:
            optimal_action = 2
        if self.soc < 20:
            optimal_action = 1
        elif self.soc > 80:
            optimal_action = 2

        if action == optimal_action:
            self.optimal_actions += 1

        if self.previous_action is not None and action != self.previous_action and action != 0 and production_diff < 0.1:
            reward -= 0.1

        if action == 1 and self.soc < 100:
            charge_amount = min(charge_rate, 100 - self.soc) * self.charge_efficiency
            self.soc += charge_amount
            production_diff = (future["future_production"] - current["energyproduced"]) / 100
            reward = 15 * production_diff if future["future_production"] > current["energyproduced"] else -1
            print(f"  Charger: Reward = {reward:.3f}")
        elif action == 2 and self.soc > 0:
            discharge_amount = min(charge_rate, self.soc) * self.discharge_efficiency
            self.soc -= discharge_amount
            production_diff = (current["energyproduced"] - future["future_production"]) / 100
            reward = 15 * production_diff if future["future_production"] < current["energyproduced"] else -1
            print(f"  Décharger: Reward = {reward:.3f}")
        else:
            production_diff = abs(future["future_production"] - current["energyproduced"]) / 100
            if production_diff > 0.05:
                reward = -0.5
            else:
                reward = 0
            print(f"  Rester: Reward = {reward:.3f}")

        if self.previous_action is not None and action != self.previous_action and action != 0:
            if (self.previous_action == 1 and action == 2) or (self.previous_action == 2 and action == 1):
                self.cycle_count += 0.5
                print(f"  Cycle count incremented: {self.cycle_count}")
        self.previous_action = action

        if self.soc < 25 or self.soc > 75:
            reward -= 0.5
            print(f"  SOC warning penalty applied: Reward = {reward:.3f}")
        if self.soc < self.soc_min or self.soc > self.soc_max:
            reward -= 2.0
            print(f"  SOC limit penalty applied: Reward = {reward:.3f}")
        elif self.soc_min + 10 <= self.soc <= self.soc_max - 10:
            reward += 1.5
            print(f"  SOC bonus applied: Reward = {reward:.3f}")

        if action == 2 and 16 <= current["hour"] <= 21:
            reward += 3.0
            print(f"  Peak hour bonus applied: Reward = {reward:.3f}")

        if (action == 1 or action == 2) and current["temperature"] > 75:
            reward -= 0.5
            print(f"  High temperature penalty applied: Reward = {reward:.3f}")

        if self.cycle_count > self.max_cycles * 0.8:
            reward -= 2
            print(f"  Cycle wear penalty applied: Reward = {reward:.3f}")

        self.index += 1
        if self.index >= self.max_index:
            done = True
            self.accuracy = (self.optimal_actions / 85) * 100
            print(f"Accuracy de l'épisode : {self.accuracy:.2f}%")

        return self._get_observation(), reward, done, {}