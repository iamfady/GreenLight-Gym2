import re
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. ملفات اللوجات
# ==============================
files = {
    "PPO": "Lettuc_ppo_output.txt",
    "RecurrentPPO": "Lettuc_recurrentppo_output.txt",
    "SAC": "Lettuc_sac_output.txt",
}

# ==============================
# 2. Regex لكل المتغيرات
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

    # States
    "temp": r"temp\s*\|\s*([0-9.eE+-]+)",
    "rh": r"rh\s*\|\s*([0-9.eE+-]+)",
    "co2": r"co2\s*\|\s*([0-9.eE+-]+)",
    "light": r"light\s*\|\s*([0-9.eE+-]+)",

    # Control actions
    "ventilation": r"ventilation\s*\|\s*([0-9.eE+-]+)",
    "heating": r"heating\s*\|\s*([0-9.eE+-]+)",
    "lamps": r"lamps\s*\|\s*([0-9.eE+-]+)",
    "thermal_screen": r"thermal_screen\s*\|\s*([0-9.eE+-]+)",
}

# ==============================
# 3. قراءة البيانات مع padding بالـ NaN
# ==============================
dataframes = {}
for model, path in files.items():
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        content = f.read()
    extracted = {k: re.findall(p, content) for k, p in patterns.items()}
    max_len = max(len(v) for v in extracted.values() if len(v) > 0)
    df = pd.DataFrame({
        k: pd.to_numeric(v + [None]*(max_len - len(v)), errors="coerce")
        for k, v in extracted.items()
    })
    df["model"] = model
    dataframes[model] = df

df_all = pd.concat(dataframes.values(), ignore_index=True)

# ==============================
# 4. دالة الرسم
# ==============================
def plot_metric(metric, cumulative=False, kind="line"):
    plt.figure(figsize=(8,5))
    for model in df_all["model"].unique():
        subset = df_all[df_all["model"]==model]
        if subset.empty or metric not in subset:
            continue
        y = subset[metric].cumsum() if cumulative else subset[metric]
        if kind == "line":
            plt.plot(subset["timesteps"], y, label=model, marker="o", alpha=0.8)
        elif kind == "scatter":
            plt.scatter(subset["timesteps"], y, label=model, alpha=0.6)
    plt.xlabel("Timesteps")
    plt.ylabel(metric)
    plt.title(f"{metric} over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(plt)

# ==============================
# 5. Streamlit UI with Tabs
# ==============================
st.title("🌱 Lettuce RL Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Learning", "💰 Costs", "⚠️ Violations", "🌱 States", "⚙️ Controls"]
)

with tab1:
    st.subheader("Learning Curves")
    plot_metric("episode_reward")
    plot_metric("EPI")

with tab2:
    st.subheader("Cumulative Costs")
    for m in ["co2_cost","heat_cost","elec_cost","variable_costs","revenue"]:
        plot_metric(m, cumulative=True)

with tab3:
    st.subheader("Cumulative Violations")
    for m in ["temp_violation","rh_violation"]:
        plot_metric(m, cumulative=True)

with tab4:
    st.subheader("Environment States")
    for m in ["temp","rh","co2","light"]:
        if m in df_all.columns and df_all[m].notna().any():
            plot_metric(m)

with tab5:
    st.subheader("Control Actions")
    for m in ["ventilation","heating","lamps","thermal_screen"]:
        if m in df_all.columns and df_all[m].notna().any():
            plot_metric(m)
