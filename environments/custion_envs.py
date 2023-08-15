import os
import yaml

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic



class Custom_Alfred_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        
        # setting alfred
        with open("alfworld/configs/base_config.yaml") as confile:
            config = yaml.load(confile, yaml.SafeLoader)
        env_type = config['env']['type']
        env = getattr(environment, env_type)(config, train_eval='train')
        self.alf_env = env.init_env(batch_size=1)
        
                
        
        
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), 
                                            dtype=np.uint8)

    def step(self, action):
        observation, scores, terminated, info = self.alf_env.step(action)
        truncated = None  # What is this?
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        
        observation, info = self.alf_env.reset()
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...
        
        
        
        
        
if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__))