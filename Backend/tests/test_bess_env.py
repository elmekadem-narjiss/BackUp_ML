import pytest
from stable_baselines3 import PPO
from BESSBatteryEnv import BESSBatteryEnv

@pytest.mark.integration
def test_bess_env_integration():
    env = BESSBatteryEnv('test_lstm_predictions.csv')
    assert env.observation_space is not None
    reset_result = env.reset()
    obs = reset_result if not isinstance(reset_result, tuple) else reset_result[0]
    assert obs is not None
    model = PPO('MlpPolicy', env, n_steps=2, batch_size=2, verbose=0)
    model.learn(total_timesteps=4)
    action, _ = model.predict(obs)
    assert action is not None
    print('Integration test passed: Model trained and predicted action:', action)
