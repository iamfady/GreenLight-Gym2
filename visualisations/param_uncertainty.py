import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plot_config

# WIDTH = 87.5 * 0.03937  # 85 mm ≈ 3.35 inches
WIDTH = 60 * 0.03937  # 180 mm ≈ 7.08 inches

HEIGHT = WIDTH * 0.75  # Adjust aspect ratio (3:2 or 4:3 is ideal)

def load_data(project, mode, algorithm, growth_year, start_day, location, model_names):
    """
    Load and organize data from CSV files based on specified parameters.
    This function reads CSV files from a hierarchical directory structure organized by project,
    mode, algorithm, and noise levels. It filters files based on growth year, start day, and
    location parameters.
    Parameters
    ----------
    project : str
        The name of the project directory
    mode : str
        The mode directory name within the project
    algorithm : str
        The algorithm directory name
    growth_year : str
        Filter parameter for the growth year in filenames
    start_day : str
        Filter parameter for the start day in filenames
    location : str
        Filter parameter for the location in filenames
    Returns
    -------
    dict
        A nested dictionary where:
        - First level keys are noise levels (as strings)
        - Values are pandas DataFrames containing the data from matching CSV files
    Notes
    -----
    - The function assumes a specific directory structure: data/project/mode/algorithm/noise_level/
    - Only processes the first matching CSV file found for each noise level
    - Noise levels are sorted numerically
    - Prints warning messages if files are not found or if no matching CSV exists
    Example
    -------
    >>> data = load_data("project1", "mode1", "algo1", "2023", "day1", "weatherlocationA")
    """
    base_path = os.path.join("data", project, mode, algorithm)
    noise_levels = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    noise_levels.sort(key=lambda x: float(x))
    print(noise_levels)
    data_dict = {}
    # Initialize dictionary to store DataFrames for each noise level and file
    data_dict = {noise_level: {} for noise_level in noise_levels}
    # Get all CSV files from the first noise level folder (assuming same files in all folders)
    for i, noise_level in enumerate(noise_levels):
        folder_path = os.path.join(base_path, noise_level)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') 
                and growth_year in f 
                and start_day in f 
                and location in f
                and model_names[i] in f.lower()]
        if csv_files:
            try:
                data_dict[noise_level] = pd.read_csv(os.path.join(folder_path, csv_files[0]))
            except FileNotFoundError:
                print(f"File not found: {os.path.join(folder_path, csv_files[0])}")
                continue
        else:
            print(f"No matching CSV file found in {folder_path}")
    return data_dict

def plot_cumulative_reward(final_metrics, col2plot, ylabel=None):

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    colors =[ "#003366", "#A60000", "grey"]
    labels = ["PPO", "SAC", "RB baseline"]
    # ax.plot(final_rewards.index, final_rewards[f"Cumulative {col2plot}"], "o-", label=col2plot)
    for i, (algorithm, final_rewards) in enumerate(final_metrics.items()):
                # ax.errorbar(final_rewards.index, final_rewards[f"Cumulative {col2plot}"], yerr=final_rewards[f"std {col2plot}"], fmt="o-", markersize=4, color=colors[i], label=algorithm.upper(), capsize=5)

        mean = final_rewards[f"Cumulative {col2plot}"]
        std = (final_rewards[f"std {col2plot}"])
        ax.plot(final_rewards.index, mean, '-', color=colors[i], label=labels[i], markersize=4)
        ax.fill_between(final_rewards.index, mean-std, mean+std, alpha=0.2, color=colors[i])
    
    ax.set_xlabel(r"Uncertainty $(\delta)$")
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(f"Cumulative {col2plot.lower()[:-1]}")
    ax.legend()
    plt.tight_layout()
    plt.show()
    # fig.savefig(f"figures/AgriControl/stochastic/{col2plot}_cumulative_reward.png")
    # fig.savefig(f"figures/AgriControl/stochastic/{col2plot}_cumulative_reward.svg", format="svg", dpi=300)

def compute_cumulative_metrics(data_dict):
    columns_to_sum = [
        'cFruit', 'Rewards', 'EPI', 'Revenue', 'Heat costs', 'CO2 costs',
        'Elec costs', 'temp_violation', 'co2_violation', 'rh_violation', 'Penalty'
    ]

    # Process each noise level's data
    for noise_level, data in data_dict.items():
        # Add a penalty column that sums all violations
        if all(col in data.columns for col in ['temp_violation', 'co2_violation', 'rh_violation']):
            data['Penalty'] = data['temp_violation'] + data['co2_violation'] + data['rh_violation']
        # Group by episode and compute cumsum within each episode
        for col in columns_to_sum:
            if col in data.columns:
                data[f'cumsum {col}'] = data.groupby('episode')[col].cumsum()

    # Create final rewards dataframe with columns for mean and std
    final_rewards = pd.DataFrame(index=data_dict.keys())
    for noise_level, data in data_dict.items():
        for col in columns_to_sum:
            if f'cumsum {col}' in data.columns:
                # Get the last value for each episode
                episode_finals = data.groupby('episode')[f'cumsum {col}'].last()
                # Calculate mean and std
                final_rewards.loc[noise_level, f'Cumulative {col}'] = episode_finals.mean()
                final_rewards.loc[noise_level, f'std {col}'] = 3.291 * episode_finals.std() / np.sqrt(len(episode_finals))

    return final_rewards

def main(args):
    algorithms = ["ppo", "sac", "rb_baseline"]
    model_names = [["hopeful-wind-295","light-wave-296","ruby-star-297","eager-resonance-298","rural-eon-300","stellar-durian-301","copper-dawn-303"],
                    ["distinctive-frost-299","stoic-moon-302","graceful-dream-304","copper-frog-305","warm-flower-306","sunny-sky-307","leafy-cloud-308"],
                    ["rb_baseline", "rb_baseline", "rb_baseline", "rb_baseline", "rb_baseline", "rb_baseline", "rb_baseline"]]
    final_metrics  = {}
    for i, algorithm in enumerate(algorithms):
        # data_dict = load_data(args.project, args.mode, args.algorithm, args.growth_year, args.start_day, args.location)
        data_dict = load_data(args.project, args.mode, algorithm, args.growth_year, args.start_day, args.location, model_names[i])
        final_rewards = compute_cumulative_metrics(data_dict)
        final_metrics[algorithm] = final_rewards

    # final_rewards = compute_cumulative_metrics(data_dict)
    plot_cumulative_reward(final_metrics, col2plot="Rewards", ylabel="Cumulative reward")
    plot_cumulative_reward(final_metrics, col2plot="EPI", ylabel=r"Cumulative EPI (EU/m$^2$)")
    plot_cumulative_reward(final_metrics, col2plot="temp_violation")
    plot_cumulative_reward(final_metrics, col2plot="co2_violation")
    plot_cumulative_reward(final_metrics, col2plot="rh_violation")
    plot_cumulative_reward(final_metrics, col2plot="Penalty", ylabel="Cumulative penalty")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cost metrics from different models")
    parser.add_argument("--project", type=str, required=True, help="Path to project folder")
    parser.add_argument("--mode", type=str, choices=["deterministic", "stochastic"], required=True, help="Simulation mode")
    parser.add_argument("--growth_year", type=str, required=True, help="Growth year")
    parser.add_argument("--start_day", type=str, required=True, help="Start day")
    parser.add_argument("--location", type=str, required=True, help="Location")
    args = parser.parse_args()

    main(args)

