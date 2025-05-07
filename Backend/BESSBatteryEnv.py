import gym
import pandas as pd
import numpy as np
from gym import spaces

class BESSBatteryEnv(gym.Env):
    def __init__(self, data_file):
        super(BESSBatteryEnv, self).__init__()
        self.data = pd.read_csv(data_file)
        # Validate required columns
        required_columns = ['energyproduced', 'demand_pred', 'actual_demand']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.soc = 0.5  # State of Charge (initially 50%)
        self.max_soc = 1.0
        self.min_soc = 0.0
        self.battery_capacity = 1000  # kWh
        self.max_charge_rate = 200  # kW
        self.max_discharge_rate = 200  # kW
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([1000, 1000, 1000, 1]),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.max_discharge_rate,
            high=self.max_charge_rate,
            shape=(1,),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.soc = 0.5
        return np.array([
            self.data['energyproduced'].iloc[0],
            self.data['demand_pred'].iloc[0],
            self.data['actual_demand'].iloc[0],
            self.soc
        ], dtype=np.float32)

    def step(self, action):
        energy_produced = self.data['energyproduced'].iloc[self.current_step]
        predicted_demand = self.data['demand_pred'].iloc[self.current_step]
        actual_demand = self.data['actual_demand'].iloc[self.current_step]
        
        # Apply action (charge/discharge)
        action = np.clip(action[0], -self.max_discharge_rate, self.max_charge_rate)
        energy_change = action / self.battery_capacity
        new_soc = np.clip(self.soc + energy_change, self.min_soc, self.max_soc)
        
        # Calculate reward
        energy_balance = energy_produced + action - actual_demand
        reward = -abs(energy_balance)  # Minimize imbalance
        if new_soc == self.min_soc or new_soc == self.max_soc:
            reward -= 10  # Penalty for hitting SOC limits
        
        self.soc = new_soc
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Prepare observation
        obs = np.array([
            energy_produced,
            predicted_demand,
            actual_demand,
            self.soc
        ], dtype=np.float32)
        
        info = {
            'energy_balance': energy_balance,
            'soc': self.soc
        }
        
        return obs, reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, SOC: {self.soc:.2f}")
