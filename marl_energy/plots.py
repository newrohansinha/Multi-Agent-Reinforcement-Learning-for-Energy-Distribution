
import json
import matplotlib.pyplot as plt
import csv

lc_csv = "outputs/training_metrics.csv"
eval_json = "outputs/eval_summary.json"
cost_plot = "outputs/cost_reduction.png"
lc_plot = "outputs/learning_curve.png"

xs = []
ys = []
with open(lc_csv,"r") as f:
    r = csv.DictReader(f)
    for row in r:
        xs.append(int(row["iteration"]))
        ys.append(float(row["episode_reward_mean"]))
plt.figure()
plt.plot(xs, ys)
plt.xlabel("Iteration")
plt.ylabel("Episode Reward Mean")
plt.title("Learning Curve")
plt.tight_layout()
plt.savefig(lc_plot)
try:
    with open(eval_json,"r") as f:
        ev = json.load(f)
    plt.figure()
    plt.bar(["Baseline","Trained"], [ev["baseline_cost"], ev["trained_cost"]])
    plt.ylabel("Average Cost")
    plt.title("Cost Reduction")
    plt.tight_layout()
    plt.savefig(cost_plot)
except Exception as e:
    pass
