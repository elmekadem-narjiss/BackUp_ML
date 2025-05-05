import gym
from gym import spaces
import numpy as np
import pandas as pd

class BESSBatteryEnv(gym.Env):
    def __init__(self, data_path='lstm_predictions_charger.csv'):
        super(BESSBatteryEnv, self).__init__()
        
        # Charger les données
        self.data = pd.read_csv(data_path)
        
        # Identifier les colonnes pour l'environnement
        self.price_column = next((col for col in ['energy_price', 'price'] if col in self.data.columns), None)
        self.demand_column = next((col for col in ['predicted_demand', 'energyproduced', 'demand'] if col in self.data.columns), None)
        
        # Vérifier la présence des colonnes nécessaires
        missing_columns = []
        if self.price_column is None:
            missing_columns.append('energy_price ou price')
        if self.demand_column is None:
            missing_columns.append('predicted_demand, energyproduced ou demand')
        if missing_columns:
            available_columns = list(self.data.columns)
            raise ValueError(
                f"Colonnes manquantes dans le CSV : {missing_columns}. "
                f"Colonnes disponibles : {available_columns}"
            )
        
        # Paramètres de la batterie
        self.capacity = 100.0  # Capacité de la batterie en kWh
        self.max_charge_rate = 50.0  # Taux de charge max en kW
        self.max_discharge_rate = 50.0  # Taux de décharge max en kW
        self.efficiency = 0.95  # Efficacité de charge/décharge
        self.state_of_charge = 0.5 * self.capacity  # État initial à 50%
        
        # Espace d'actions : charge (>0) ou décharge (<0) en kW
        self.action_space = spaces.Box(
            low=-self.max_discharge_rate,
            high=self.max_charge_rate,
            shape=(1,),
            dtype=np.float32
        )
        
        # Espace d'observations : [state_of_charge, energy_price, predicted_demand]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([self.capacity, np.inf, np.inf]),
            shape=(3,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(self.data) - 1

    def reset(self):
        self.current_step = 0
        self.state_of_charge = 0.5 * self.capacity
        return self._get_observation()

    def step(self, action):
        action = action[0]
        prev_soc = self.state_of_charge
        
        # Calculer la nouvelle charge
        if action > 0:  # Charge
            charge = min(action * self.efficiency, self.max_charge_rate, self.capacity - self.state_of_charge)
            self.state_of_charge += charge
        else:  # Décharge
            discharge = min(-action / self.efficiency, self.max_discharge_rate, self.state_of_charge)
            self.state_of_charge -= discharge
        
        # Avancer le step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Calculer la récompense
        reward = self._calculate_reward(action)
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Utiliser une valeur par défaut pour energy_price si la colonne est absente
        energy_price = self.data.iloc[self.current_step][self.price_column] if self.price_column else 0.1
        predicted_demand = self.data.iloc[self.current_step][self.demand_column]
        return np.array([self.state_of_charge, energy_price, predicted_demand], dtype=np.float32)

    def _calculate_reward(self, action):
        # Utiliser une valeur par défaut pour energy_price si la colonne est absente
        energy_price = self.data.iloc[self.current_step][self.price_column] if self.price_column else 0.1
        predicted_demand = self.data.iloc[self.current_step][self.demand_column]
        
        # Récompense basée sur le coût et la satisfaction de la demande
        cost = action * energy_price
        demand_error = abs(predicted_demand - action)
        reward = -cost - 0.1 * demand_error
        
        return reward

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, SoC: {self.state_of_charge}, Action: {self.action_space.sample()[0]}")
