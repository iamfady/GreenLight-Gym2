import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import cmcrameri.cm as cmc

import plot_config

plt.rc("axes", unicode_minus=False)

WIDTH = 87.5 * 0.03937  # 85 mm ≈ 3.35 inches
HEIGHT = WIDTH * 0.75  # Adjust aspect ratio (3:2 or 4:3 is ideal)

def convert_to_fps(times):
    n_steps = 10 * 24 * 12
    time_per_step = times / n_steps
    return 1 / time_per_step

# Read the CSV files
cpp_times = pd.read_csv("data/AgriControl/run_times/gl_gym.csv")
glplus_times = pd.read_csv("data/AgriControl/run_times/glplus.csv")
gl_matlab_times = pd.read_csv("data/AgriControl/run_times/gl_matlab.csv")

cpp_fps = convert_to_fps(cpp_times["elapsed_time"])
glplus_fps = convert_to_fps(glplus_times["elapsed_time"])
gl_matlab_fps = convert_to_fps(gl_matlab_times["elapsed_time"])

# Calculate means and standard deviations
cpp_mean = cpp_fps.mean()
cpp_std = cpp_fps.std()
glplus_mean = glplus_fps.mean()
glplus_std = glplus_fps.std()
gl_matlab_mean = gl_matlab_fps.mean()
gl_matlab_std = gl_matlab_fps.std()


# Create bar plot
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
models = ["GL-Matlab", "GL-Python", "GL-Gym"]
means = [gl_matlab_mean, glplus_mean, cpp_mean]
stds = [gl_matlab_std, glplus_std,cpp_std ]

cpp_color = "C0"
glplus_color = "#A60000"
gl_matlab_color = "grey"
colors = [gl_matlab_color, glplus_color, cpp_color]

bars = ax.bar(models, means, yerr=stds, color=colors, alpha=1)

# Customize plot
ax.set_ylabel("Steps per second")
ax.set_yticks([0, 500, 1000, 1500, 2000])
ax.set_ylim(0, 2000)
plt.tight_layout()
print(means)
print("factor of speed up over GL Matlab: ", cpp_mean/gl_matlab_mean)
print("factor of speed up over GL+: ", cpp_mean/glplus_mean)
# plt.savefig('figures/AgriControl/run_times/fps.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.savefig('figures/AgriControl/run_times/fps.png', format='png', bbox_inches='tight')

plt.show()