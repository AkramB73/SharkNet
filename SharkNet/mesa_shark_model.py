import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

DEG_LAT_TO_M = 111132.0

class SharkAgent(Agent):
    # unique_id is automatically assigned
    def __init__(self, model, energy=100.0, preferred_temp=(8.0, 18.0), rl_policy=None):
        super().__init__(model)
        self.energy = energy
        self.preferred_temp = preferred_temp
        self.rl_policy = rl_policy
        self.pos_float = (0.0, 0.0)

    def sense(self):
        x_int, y_int = int(self.pos_float[0]), int(self.pos_float[1])
        neighborhood = self.model.get_neighborhood_values(x_int, y_int, radius=1)
        return {'pos': (x_int, y_int), 'energy': self.energy, 'neighborhood': neighborhood}

    def choose_action(self, obs):
        if callable(self.rl_policy):
            return self.rl_policy(obs)
            
        # heuristic fallback for no RL policy (testing)
        best_move, best_score = (0, 0), -np.inf
        for (dx, dy), values in obs['neighborhood'].items():
            score = values['chl'] - abs(values['sst'] - np.mean(self.preferred_temp))
            if score > best_score:
                best_score, best_move = score, (dx, dy)
        return best_move

    def step(self):
        obs = self.sense()
        action = self.choose_action(obs)
        self.model.execute_action(self, action)

class SharkModel(Model):
    def __init__(self, width, height, curr_u, curr_v, chl, sst, n_agents=1, time_step_hrs=0.12, seed=None):
        super().__init__(seed=seed)
        self.width, self.height = width, height
        self.grid = MultiGrid(width, height, torus=False)
        self.curr_u, self.curr_v = curr_u, curr_v
        self.food_layer, self.temp_layer = chl, sst
        
        self.time_step_secs = time_step_hrs * 3600
        self.cell_meters = DEG_LAT_TO_M * 0.1 

        self.metabolic_cost = 0.5
        self.food_gain_factor = 1.0

        for _ in range(n_agents):
            a = SharkAgent(self)
            x = self.random.random() * self.width
            y = self.random.random() * self.height
            a.pos_float = (x, y)
            self.grid.place_agent(a, (int(x), int(y)))

    def get_neighborhood_values(self, x, y, radius=1):
        vals = {}
        for nx, ny in self.grid.get_neighborhood((x, y), moore=True, include_center=True, radius=radius):
            # Grid coords:(col, row), arrays coords:[row, col]
            vals[(nx - x, ny - y)] = {'chl': self.food_layer[ny, nx], 'sst': self.temp_layer[ny, nx]}
        return vals

    def execute_action(self, agent, action):
        dx, dy = action
        x_f, y_f = agent.pos_float
        
        x_f += dx
        y_f += dy
        
        x_i, y_i = int(agent.pos_float[0]), int(agent.pos_float[1])

        # advection currents (u: eastward, v: northward)
        u, v = self.curr_u[y_i, x_i], self.curr_v[y_i, x_i]
        x_f += (u * self.time_step_secs) / self.cell_meters
        y_f -= (v * self.time_step_secs) / self.cell_meters
        # Boundary conditions
        x_f = max(0, min(self.width - 1e-6, x_f))
        y_f = max(0, min(self.height - 1e-6, y_f))

        agent.pos_float = (x_f, y_f)
        self.grid.move_agent(agent, (int(x_f), int(y_f)))

        food_at_pos = self.food_layer[int(y_f), int(x_f)]
        agent.energy += (food_at_pos * self.food_gain_factor) - self.metabolic_cost

    def step(self):
        self.agents.shuffle_do("step")