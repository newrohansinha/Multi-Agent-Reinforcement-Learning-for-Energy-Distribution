
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class EnergyMAEnv(MultiAgentEnv):
    metadata = {"render_modes": []}
    def __init__(self, config):
        self.cfg = config or {}
        self.num_homes = int(self.cfg.get("num_homes", 3))
        self.T = int(self.cfg.get("episode_len", 48))
        self.price_low = float(self.cfg.get("price_low", 1.0))
        self.price_high = float(self.cfg.get("price_high", 3.0))
        self.peak_start = int(self.cfg.get("peak_start", 24))
        self.peak_end = int(self.cfg.get("peak_end", 40))
        self.flex_prob = float(self.cfg.get("flex_prob", 0.35))
        self.base_min = float(self.cfg.get("base_min", 0.6))
        self.base_max = float(self(self.cfg.get("base_max", 1.2)))
        self.flex_unit = float(self.cfg.get("flex_unit", 0.8))
        self.flex_backlog_limit = int(self.cfg.get("flex_backlog_limit", 6))
        self.battery_capacity = float(self.cfg.get("battery_capacity", 12.0))
        self.battery_init = float(self.cfg.get("battery_init", 6.0))
        self.charge_rate = float(self.cfg.get("charge_rate", 2.0))
        self.discharge_rate = float(self.cfg.get("discharge_rate", 2.0))
        self.efficiency = float(self.cfg.get("efficiency", 0.95))
        self.grid_capacity = float(self.cfg.get("grid_capacity", 6.0))
        self.unmet_penalty = float(self.cfg.get("unmet_penalty", 10.0))
        self.backlog_penalty = float(self.cfg.get("backlog_penalty", 0.02))
        self.surge_factor = float(self.cfg.get("surge_factor", 1.0))
        self.seed_val = int(self.cfg.get("seed", 42))
        self.rng = np.random.default_rng(self.seed_val)
        self.home_ids = [f"home_{i}" for i in range(self.num_homes)]
        self.battery_id = "battery"
        self.agent_ids = self.home_ids + [self.battery_id]
        self.base_demand = np.zeros(self.num_homes, dtype=np.float32)
        self.flex_demand = np.zeros(self.num_homes, dtype=np.float32)
        self.backlog = np.zeros(self.num_homes, dtype=np.float32)
        self.soc = 0.0
        self.t = 0
        self.total_cost = 0.0
        self.total_unmet = 0.0
        self.price_level = 0
        self.price = self.price_low
        home_low = np.array([0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float32)
        home_high = np.array([1.0,1.0,2.0,2.0,10.0,1.0], dtype=np.float32)
        self.home_obs_space = spaces.Box(low=home_low, high=home_high, dtype=np.float32)
        self.home_act_space = spaces.Discrete(3)
        batt_low = np.array([0.0,0.0,6.0,1.0], dtype=np.float32)
        batt_high = np.array([1.0,1.0,20.0,1.0], dtype=np.float32)
        self.batt_obs_space = spaces.Box(low=batt_low, high=batt_high, dtype=np.float32)
        self.batt_act_space = spaces.Discrete(3)

    def observation_space_sample(self, agent_id):
        if agent_id == self.battery_id:
            return self.batt_obs_space.sample()
        return self.home_obs_space.sample()

    def observation_space(self, agent_id=None):
        if agent_id is None:
            return {a:(self.batt_obs_space if a==self.battery_id else self.home_obs_space) for a in self.agent_ids}
        return self.batt_obs_space if agent_id==self.battery_id else self.home_obs_space

    def action_space(self, agent_id=None):
        if agent_id is None:
            return {a:(self.batt_act_space if a==self.battery_id else self.home_act_space) for a in self.agent_ids}
        return self.batt_act_space if agent_id==self.battery_id else self.home_act_space

    def price_at(self, t):
        return self.price_high if self.peak_start <= t < self.peak_end else self.price_low

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.soc = float(self.battery_init)
        self.backlog[:] = 0.0
        self.total_cost = 0.0
        self.total_unmet = 0.0
        self.base_demand = self.rng.uniform(self.base_min, self.base_max, size=self.num_homes).astype(np.float32)
        self.flex_demand = (self.rng.random(self.num_homes) < self.flex_prob).astype(np.float32) * self.flex_unit
        self.price_level = 1 if self.peak_start <= 0 < self.peak_end else 0
        self.price = self.price_at(0)
        obs = {}
        for i,h in enumerate(self.home_ids):
            obs[h] = np.array([self.price_level, self.soc/self.battery_capacity, self.base_demand[i], self.flex_demand[i], self.backlog[i], self.t/self.T], dtype=np.float32)
        total_demand = float(np.sum(self.base_demand + self.flex_demand + self.backlog))
        obs[self.battery_id] = np.array([self.price_level, self.soc/self.battery_capacity, total_demand, self.t/self.T], dtype=np.float32)
        infos = {a:{} for a in self.agent_ids}
        return obs, infos

    def step(self, action_dict):
        for a in self.agent_ids:
            if a not in action_dict:
                action_dict[a] = 0
        serve_flex = np.zeros(self.num_homes, dtype=np.float32)
        for i,h in enumerate(self.home_ids):
            act = int(action_dict[h])
            if act == 0:
                serve_flex[i] = self.flex_demand[i] + self.backlog[i]
                self.backlog[i] = 0.0
            elif act == 1:
                self.backlog[i] = min(self.backlog[i] + self.flex_demand[i], self.flex_backlog_limit)
                serve_flex[i] = 0.0
            elif act == 2:
                serve_flex[i] = self.flex_demand[i]
            else:
                serve_flex[i] = 0.0
        total_base = float(np.sum(self.base_demand))
        total_flex = float(np.sum(serve_flex))
        total_backlog_now = float(np.sum(self.backlog))
        battery_act = int(action_dict[self.battery_id])
        discharge = 0.0
        charge = 0.0
        if battery_act == 1:
            charge = min(self.charge_rate, self.battery_capacity - self.soc)
        elif battery_act == 2:
            discharge = min(self.discharge_rate, self.soc)
        effective_discharge = discharge * self.efficiency
        demand = (total_base + total_flex) * self.surge_factor
        supply_from_batt = min(effective_discharge, demand)
        grid_needed = demand - supply_from_batt + charge
        grid_supply = min(grid_needed, self.grid_capacity)
        unmet = max(0.0, grid_needed - grid_supply)
        self.soc = min(self.battery_capacity, max(0.0, self.soc + charge - discharge))
        price = self.price_at(self.t)
        cost = price * grid_supply
        delay_penalty = self.backlog_penalty * total_backlog_now
        step_cost = cost + self.unmet_penalty * unmet + delay_penalty
        self.total_cost += cost
        self.total_unmet += unmet
        rewards = {a: -step_cost for a in self.agent_ids}
        self.t += 1
        done = self.t >= self.T
        if not done:
            self.flex_demand = (self.rng.random(self.num_homes) < self.flex_prob).astype(np.float32) * self.flex_unit
            self.base_demand = self.rng.uniform(self.base_min, self.base_max, size=self.num_homes).astype(np.float32)
        self.price_level = 1 if self.peak_start <= self.t < self.peak_end else 0
        self.price = self.price_at(self.t if self.t < self.T else self.T-1)
        obs = {}
        for i,h in enumerate(self.home_ids):
            obs[h] = np.array([self.price_level, self.soc/self.battery_capacity, self.base_demand[i] if not done else 0.0, self.flex_demand[i] if not done else 0.0, self.backlog[i], (self.t if not done else self.T)/self.T], dtype=np.float32)
        total_demand_next = float(np.sum(self.base_demand + self.flex_demand + self.backlog)) if not done else 0.0
        obs[self.battery_id] = np.array([self.price_level, self.soc/self.battery_capacity, total_demand_next, (self.t if not done else self.T)/self.T], dtype=np.float32)
        terminateds = {a: done for a in self.agent_ids}
        truncateds = {a: False for a in self.agent_ids}
        terminateds["__all__"] = done
        truncateds["__all__"] = False
        infos = {a: {"cost": cost, "unmet": unmet} for a in self.agent_ids}
        return obs, rewards, terminateds, truncateds, infos
