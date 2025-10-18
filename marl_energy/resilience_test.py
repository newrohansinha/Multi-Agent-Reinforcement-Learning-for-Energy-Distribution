
import argparse, json, yaml, os
import ray
import matplotlib.pyplot as plt
from marl_energy.utils.rollout import rollout
from marl_energy.baseline import run_baseline
from marl_energy.envs.energy_env import EnergyMAEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

def env_creator(cfg):
    return EnergyMAEnv(cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--surge", type=float, default=2.0)
    parser.add_argument("--cfg", type=str, default="config/config.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoint")
    parser.add_argument("--out", type=str, default="outputs/resilience_summary.json")
    args = parser.parse_args()
    with open(args.cfg,"r") as f:
        env_cfg = yaml.safe_load(f)
    register_env("energy_ma", lambda c: env_creator(c))
    dummy = EnergyMAEnv(env_cfg)
    policies = {
        "home_policy": (None, dummy.home_obs_space, dummy.home_act_space, {}),
        "battery_policy": (None, dummy.batt_obs_space, dummy.batt_act_space, {}),
    }
    def map_fn(agent_id, episode=None, worker=None, **kwargs):
        return "battery_policy" if agent_id.startswith("battery") else "home_policy"
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    algo = (
        PPOConfig()
        .framework("torch")
        .environment(env="energy_ma", env_config=env_cfg)
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
        .multi_agent(policies=policies, policy_mapping_fn=map_fn)
    ).build()
    algo.restore(args.ckpt)
    t_cost, t_unmet = rollout(algo, env_cfg, episodes=args.episodes, surge=args.surge)
    b_cost, b_unmet = run_baseline(env_cfg, episodes=args.episodes, surge=args.surge)
    res = {"episodes": args.episodes, "surge_factor": args.surge, "baseline_unmet": b_unmet, "trained_unmet": t_unmet, "baseline_cost": b_cost, "trained_cost": t_cost}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f:
        import json
        json.dump(res, f, indent=2)
    try:
        plt.figure()
        plt.bar(["Baseline Unmet","Trained Unmet"], [b_unmet, t_unmet])
        plt.ylabel("Avg Unmet Demand")
        plt.title("Resilience Under Demand Surge")
        plt.tight_layout()
        plt.savefig("outputs/resilience.png")
    except Exception:
        pass
    ray.shutdown()

if __name__ == "__main__":
    main()
