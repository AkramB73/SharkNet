import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mesa_shark_model import SharkModel

class SharkGymEnv(gym.Env):    
    def __init__(self, processed_maps_dir):
        super().__init__()
        
        self.map_files = [os.path.join(processed_maps_dir, f) for f in os.listdir(processed_maps_dir) if f.endswith('.npz')]
        if not self.map_files:
            raise ValueError(f"No map files (.npz) found in directory: {processed_maps_dir}")

        self.model = None
        self.episode_steps = 0
        self.max_episode_steps = 200

        self.action_space = spaces.Discrete(9)
        self._action_to_direction = {
            0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0), 4: (1, -1),
            5: (-1, 0), 6: (-1, -1), 7: (0, -1), 8: (-1, 1),
        }
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32) # 9 chl + 9 sst + 1 energy

    def _get_obs(self):
        agent = self.model.agents[0]
        
        neighborhood = self.model.get_neighborhood_values(int(agent.pos_float[0]), int(agent.pos_float[1]), radius=1)
        
        chl_hood = []
        sst_hood = []
        
        default_vals = {'chl': 0.0, 'sst': 0.0}
        for dy in [1, 0, -1]:
            for dx in [-1, 0, 1]:
                cell_data = neighborhood.get((dx, dy), default_vals)
                chl_hood.append(cell_data['chl'])
                sst_hood.append(cell_data['sst'])

        chl_hood = np.array(chl_hood)
        sst_hood = np.array(sst_hood)

        # Normalise obs to [-1, 1]
        norm_chl = np.clip(chl_hood / 5.0, 0, 1) * 2 - 1
        norm_sst = np.clip((sst_hood - 5) / 30.0, 0, 1) * 2 - 1
        norm_energy = np.clip(agent.energy / 100.0, 0, 1) * 2 - 1
        
        obs = np.concatenate([norm_chl, norm_sst, [norm_energy]]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        random_map_file = self.np_random.choice(self.map_files)
        env_data = np.load(random_map_file)
        u, v, chl, sst = env_data['u'], env_data['v'], env_data['chl'], env_data['sst']
        
        height, width = u.shape
        self.model = SharkModel(width, height, u, v, chl, sst, n_agents=1, seed=seed)
        
        self.episode_steps = 0
        
        return self._get_obs(), {}

    def step(self, action):
        action_int = int(action)

        direction = self._action_to_direction[action_int]
        agent = self.model.agents[0]
        
        energy_before = agent.energy
        y_pos, x_pos = int(agent.pos_float[1]), int(agent.pos_float[0])
        temp_before = self.model.temp_layer[y_pos, x_pos]
        
        self.model.execute_action(agent, direction)
        
        energy_after = agent.energy
        energy_gained = energy_after - energy_before
        
        reward = energy_gained
        t_min, t_max = agent.preferred_temp
        if (t_min <= temp_before <= t_max):
            reward -= 1.0
        if (t_min - 5.0 <= temp_before <= t_min) or (t_max <= temp_before <= t_max + 5.0):
            reward -= 2.0
        if not(t_min - 5.0 <= temp_before <= t_max + 5.0):
            reward -= 4.0
        if action_int != 0:
            reward -= 0.1
            
        self.episode_steps += 1
        done = agent.energy <= 0
        truncated = self.episode_steps >= self.max_episode_steps
        
        return self._get_obs(), reward, done, truncated, {}