import re
import os
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1. الملفات
# ==============================
files = {
    "PPO": "Lettuc_ppo_output.txt",
    "RecurrentPPO": "Lettuc_recurrentppo_output.txt",
    "SAC": "Lettuc_sac_output.txt",
}

# فولدر لحفظ الرسومات
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

# ==============================
# 2. Regex لاستخراج القيم
# ==============================
patterns = {
    "timesteps": r"num_timesteps=([0-9]+)",
    "episode_reward": r"episode_reward=([0-9.]+)",
    "EPI": r"EPI\s*\|\s*([-0-9.]+)",
    "co2_cost": r"co2_cost\s*\|\s*([0-9.eE+-]+)",
    "elec_cost": r"elec_cost\s*\|\s*([0-9.eE+-]+)",
    "heat_cost": r"heat_cost\s*\|\s*([0-9.eE+-]+)",
    "revenue": r"revenue\s*\|\s*([0-9.eE+-]+)",
    "rh_violation": r"rh_violation\s*\|\s*([0-9.eE+-]+)",
    "temp_violation": r"temp_violation\s*\|\s*([0-9.eE+-]+)",
    "variable_costs": r"variable_costs\s*\|\s*([0-9.eE+-]+)",
}

# ==============================
# 3. قراءة الملفات + بناء DataFrame
# ==============================
dataframes = {}
for model, path in files.items():
    with open(path, "r") as f:
        content = f.read()
    extracted = {k: re.findall(p, content) for k, p in patterns.items()}
    df = pd.DataFrame({k: pd.to_numeric(v, errors="coerce") for k, v in extracted.items()})
    df["model"] = model
    dataframes[model] = df

df_all = pd.concat(dataframes.values(), ignore_index=True)

# ==============================
# 4. دالة عامة للرسم + الحفظ
# ==============================
def plot_metric(metric, ylabel=None, log_scale=False, cumulative=False):
    plt.figure(figsize=(8,5))
    for model in df_all['model'].unique():
        subset = df_all[df_all['model'] == model]
        values = subset[metric].cumsum() if cumulative else subset[metric]
        plt.plot(subset['timesteps'], values, marker='o', label=model)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel if ylabel else metric)
    plt.title(f"{metric} over Time")
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # حفظ الصور
    fname = f"{metric}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.savefig(os.path.join(out_dir, f"{metric}.svg"), format="svg")
    print(f"[✔] Saved: {fname}")
    plt.close()

# ==============================
# 5. الرسومات
# ==============================
# منحنى التعلم
plot_metric("episode_reward", "Episode Reward")

# Costs
for m in ["co2_cost", "heat_cost", "elec_cost", "variable_costs"]:
    plot_metric(m, ylabel="Cost", cumulative=True)

# Violations
for m in ["rh_violation", "temp_violation"]:
    plot_metric(m, ylabel="Violation", cumulative=True, log_scale=True)

# Revenue + EPI
plot_metric("revenue", "Revenue", cumulative=True)
plot_metric("EPI", "EPI")
