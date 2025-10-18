
import argparse, os, csv, yaml
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from marl_energy.envs.energy_env import EnergyMAEnv

def env_creator(cfg):
    return EnergyMAEnv(cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--homes", type=int, default=3)
    parser.add_argument("--out", type=str, default="outputs/training_metrics.csv")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoint")
    parser.add_argument("--cfg", type=str, default="config/config.yaml")
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["num_homes"] = args.homes
    register_env("energy_ma", lambda c: env_creator(c))
    dummy = EnergyMAEnv(base_cfg)
    policies = {
        "home_policy": (None, dummy.home_obs_space, dummy.home_act_space, {}),
        "battery_policy": (None, dummy.batt_obs_space, dummy.batt_act_space, {}),
    }
    def map_fn(agent_id, episode=None, worker=None, **kwargs):
        return "battery_policy" if agent_id.startswith("battery") else "home_policy"
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    cfg = (
        PPOConfig()
        .framework("torch")
        .environment(env="energy_ma", env_config=base_cfg)
        .rollouts(num_rollout_workers=0)
        .training(sgd_minibatch_size=256, train_batch_size=4096, lr=5e-4)
        .resources(num_gpus=0)
        .multi_agent(policies=policies, policy_mapping_fn=map_fn)
    )
    algo = cfg.build()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration","episode_reward_mean"])
        for i in range(args.iters):
            r = algo.train()
            w.writerow([i+1, r.get("episode_reward_mean", 0.0)])
    os.makedirs(args.ckpt, exist_ok=True)
    algo.save(args.ckpt)
    ray.shutdown()

if __name__ == "__main__":
    main()
