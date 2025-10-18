# Technical Write-Up

## Environment
The system models H homes and one shared battery across T steps. Each step has a binary price level. Homes have base and flexible demand; flexible demand can be deferred. The battery has capacity, charge/discharge power limits, and round-trip efficiency. The grid has a capacity cap; demand over the cap becomes unmet.

State for a home includes price level, battery state of charge, base demand, flexible demand, backlog, and a normalized time index. State for the battery includes price level, battery state, aggregate demand, and time index. Actions are discrete: homes choose to serve all, defer flexible load, or request battery; the battery chooses idle, charge, or discharge.

Reward is the negative of cost plus penalties for unmet demand and growing backlog. All agents receive a shared team reward, yielding cooperation.

## MARL Setup
RLlib PPO is configured with two policies: a home policy and a battery policy, mapped by agent id. PPO uses PyTorch and Generalized Advantage Estimation. Episodes span T steps. Training logs episode reward means to CSV.

## Baseline
The baseline serves all demand immediately and charges the battery when not full, never discharging for peak shaving. This creates higher costs during peak periods and low resilience to surges.

## Metrics
Cost is energy bought from the grid times the step price. The project reports percentage reduction in total cost relative to the baseline and plots learning curves. Resilience is tested by doubling demand for a window and comparing unmet demand and cost for trained vs baseline.

## Results Artifacts
- `outputs/training_metrics.csv` and `outputs/learning_curve.png`
- `outputs/eval_summary.json` and `outputs/cost_reduction.png`
- `outputs/resilience_summary.json` and `outputs/resilience.png`
