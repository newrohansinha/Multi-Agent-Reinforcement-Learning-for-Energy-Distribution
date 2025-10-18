
import numpy as np
from marl_energy.envs.energy_env import EnergyMAEnv

def baseline_action(env, obs):
    acts = {}
    for aid in env.home_ids:
        acts[aid] = 0
    if env.soc < env.battery_capacity:
        acts[env.battery_id] = 1
    else:
        acts[env.battery_id] = 0
    return acts

def run_baseline(env_config, episodes=5, surge=1.0, seed=123):
    cfg = dict(env_config)
    cfg["surge_factor"] = surge
    env = EnergyMAEnv(cfg)
    rng = np.random.default_rng(seed)
    costs = []
    unmeters = []
    for ep in range(episodes):
        obs,_ = env.reset(seed=rng.integers(0, 10**9))
        done = False
        while not done:
            acts = baseline_action(env, obs)
            obs, r, term, trunc, info = env.step(acts)
            done = term["__all__"] or trunc["__all__"]
        costs.append(env.total_cost)
        unmeters.append(env.total_unmet)
    return float(np.mean(costs)), float(np.mean(unmeters))
