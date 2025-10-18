# Multi-Agent Reinforcement Learning for Energy Distribution

**GitHub:** [https://github.com/newrohansinha/Multi-Agent-Reinforcement-Learning-for-Energy-Distribution.git](https://github.com/newrohansinha/Multi-Agent-Reinforcement-Learning-for-Energy-Distribution.git)

---

Multi-Agent Reinforcement Learning for energy grids. Python + Ray/RLlib (PPO), PyTorch, Gym, Matplotlib. Custom env: 2–3 home agents + 1 battery, peak/off-peak pricing. Trains to 15–20% cost drop vs baseline; CSV/PNG learning curves. Resilience test: handles 2× demand surge with lower unmet load.

---

## Overview
This project simulates an energy grid where multiple autonomous agents learn to cooperatively manage power distribution. Homes and a shared battery agent optimize energy use under fluctuating price conditions. The MARL setup achieves quantifiable cost reduction and improved resilience to demand spikes.

**Key Highlights**
- Multi-agent PPO training using **Ray RLlib + PyTorch**
- **Custom Gymnasium environment** with 3 homes + 1 battery
- Tracks cost, unmet demand, and backlog penalties
- Demonstrates **15–20% energy cost reduction**
- **2× demand surge** resilience test showing lower unmet load
- Generates CSV logs and PNG plots for convergence and comparisons

---

## Tech Stack
- **Python 3.9+**
- **Ray [RLlib]**
- **PyTorch**
- **Gymnasium**
- **NumPy**
- **Matplotlib**
- **PyYAML**

---

## Installation
```bash
git clone https://github.com/newrohansinha/Multi-Agent-Reinforcement-Learning-for-Energy-Distribution.git
cd Multi-Agent-Reinforcement-Learning-for-Energy-Distribution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`
```
ray[rllib]
torch
gymnasium
numpy
matplotlib
pyyaml
```

---

## Folder Structure
```
marl-energy/
├── marl_energy/
│   ├── envs/
│   │   └── energy_env.py
│   ├── utils/
│   │   └── rollout.py
│   ├── baseline.py
│   ├── train.py
│   ├── eval.py
│   ├── plots.py
│   └── resilience_test.py
├── config/
│   └── config.yaml
├── outputs/
├── writeup.md
└── README.md
```

---

## Environment Design
- **Agents:** Home_0..N and Battery
- **Actions (Homes):** Serve flexible load, defer, or request battery  
- **Actions (Battery):** Idle, charge, discharge  
- **Observation:** Price, SoC, demand, backlog, time index  
- **Reward:** Negative total cost + penalties (unmet, backlog)
- **Pricing Model:** Binary (peak/off-peak)
- **Constraints:** Battery efficiency, capacity, grid limits

Configuration: `config/config.yaml`

---

## Training
Train PPO for cooperative cost minimization:
```bash
python marl_energy/train.py --iters 150 --homes 3
```
Outputs:
- `outputs/training_metrics.csv`
- `outputs/checkpoint/`

---

## Evaluation
Compare trained vs baseline:
```bash
python marl_energy/eval.py --episodes 20
```
Produces:
- `outputs/eval_summary.json`

Example JSON:
```json
{
  "episodes": 20,
  "baseline_cost": 112.7,
  "trained_cost": 92.3,
  "cost_reduction_frac": 0.181,
  "baseline_unmet": 2.3,
  "trained_unmet": 0.7
}
```

---

## Visualization
Generate learning curves and cost plots:
```bash
python marl_energy/plots.py
```
Outputs:
- `outputs/learning_curve.png`
- `outputs/cost_reduction.png`

---

## Resilience Test
Simulate 2× demand surge:
```bash
python marl_energy/resilience_test.py --episodes 10 --surge 2.0
```
Outputs:
- `outputs/resilience_summary.json`
- `outputs/resilience.png`

---

## Baseline Policy
- Homes serve all flexible load immediately  
- Battery charges when not full  
- Never discharges proactively during peak  
Used to benchmark the learned policy.

---

## Expected Results
| Metric | Baseline | Trained |
|:-------|:----------|:--------|
| Avg Cost | High | 15–20% lower |
| Unmet Demand | High under surge | Significantly lower |
| Reward Curve | Unstable | Convergent & steady |

---

## Notes
- Adjust config parameters to scale episode length, price windows, and demand variation.
- CPU-only by default; GPU optional via Ray config.
- All logs, checkpoints, and plots saved to `outputs/`.

---

