import pandas as pd
import numpy as np
from gym import Env
from gym.spaces import Box

def fetch_lstm_predictions_from_drive(file_path):
    """Lit les prédictions LSTM à partir d'un fichier CSV local."""
    try:
        df = pd.read_csv(file_path)
        # Exclure explicitement les colonnes de type timestamp ou non numériques
        exclude_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()]
        df = df.drop(columns=exclude_cols, errors='ignore')
        # Sélectionner uniquement les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError(f"Aucune colonne numérique trouvée dans {file_path}. Colonnes disponibles : {df.columns}")
        df_numeric = df[numeric_cols]
        return df_numeric
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture de {file_path}: {str(e)}")

class BESSBatteryEnv(Env):
    def __init__(self, file_path):
        super(BESSBatteryEnv, self).__init__()
        self.data = fetch_lstm_predictions_from_drive(file_path)
        self.current_step = 0
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Mettre à jour l'espace d'observation en fonction du nombre de colonnes numériques
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(self.data.columns),), dtype=np.float32)
        self.max_steps = len(self.data) - 1

    def reset(self):
        self.current_step = 0
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Observation non numérique ou invalide dans reset : {obs}")
        return obs

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        reward = self._calculate_reward(action)
        obs = self.data.iloc[self.current_step].values.astype(np.float32) if not done else np.zeros(len(self.data.columns), dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Observation non numérique ou invalide dans step : {obs}")
        return obs, reward, done, {}

    def _calculate_reward(self, action):
        # Logique de calcul de la récompense (à adapter selon votre implémentation)
        return 0.0  # Placeholder

    def render(self):
        pass
