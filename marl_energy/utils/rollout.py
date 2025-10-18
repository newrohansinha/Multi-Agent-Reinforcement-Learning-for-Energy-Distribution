
import numpy as np
from marl_energy.envs.energy_env import EnergyMAEnv

def policy_map_fn(agent_id):
    return "battery_policy" if agent_id.startswith("battery") else "home_policy"

def rollout(alg, env_config, episodes=5, surge=1.0, seed=999):
    cfg = dict(env_config)
    cfg["surge_factor"] = surge
    env = EnergyMAEnv(cfg)
    rng = np.random.default_rng(seed)
    costs = []
    unmeters = []
    for ep in range(episodes):
        obs,_ = env.reset(seed=rng.integers(0, 10**9))
        states = {}
        done = False
        while not done:
            acts = {}
            for aid, ob in obs.items():
                pol = policy_map_fn(aid)
                a, s, _ = alg.get_policy(pol).compute_single_action(ob, state=states.get(aid, None), explore=False)
                states[aid] = s
                acts[aid] = a
            obs, r, term, trunc, info = env.step(acts)
            done = term["__all__"] or trunc["__all__"]
        costs.append(env.total_cost)
        unmeters.append(env.total_unmet)
    return float(np.mean(costs)), float(np.mean(unmeters))
