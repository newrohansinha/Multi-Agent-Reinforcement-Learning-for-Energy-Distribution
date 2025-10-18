
import argparse, json, yaml, os
import ray
from marl_energy.utils.rollout import rollout
from marl_energy.baseline import run_baseline
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from marl_energy.envs.energy_env import EnergyMAEnv

def env_creator(cfg):
    return EnergyMAEnv(cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--cfg", type=str, default="config/config.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoint")
    parser.add_argument("--out", type=str, default="outputs/eval_summary.json")
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
    trained_cost, trained_unmet = rollout(algo, env_cfg, episodes=args.episodes, surge=1.0)
    base_cost, base_unmet = run_baseline(env_cfg, episodes=args.episodes, surge=1.0)
    red = (base_cost - trained_cost)/max(1e-9, base_cost)
    out = {
        "episodes": args.episodes,
        "baseline_cost": base_cost,
        "trained_cost": trained_cost,
        "cost_reduction_frac": red,
        "baseline_unmet": base_unmet,
        "trained_unmet": trained_unmet
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f:
        json.dump(out, f, indent=2)
    ray.shutdown()

if __name__ == "__main__":
    main()
