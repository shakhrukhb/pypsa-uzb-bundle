#!/usr/bin/env python3
"""
PyPSA Uzbekistan - Result Visualization
========================================
Emulating the Berlin Economics PyPSA style.
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIG
PROJECT_DIR = "/home/node/projects/pypsa-uzb"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR = os.path.join(PROJECT_DIR, "results/plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define standard colors (Berlin Economics Style)
COLORS = {
    "gas": "#ff7f0e",      # Orange
    "coal": "#7f7f7f",     # Grey
    "hydro": "#1f77b4",    # Blue
    "solar": "#ffcc00",    # Yellow
    "oil": "#8c564b",      # Brown
    "AC": "#aec7e8",       # Light Blue (Imports)
    "loadshedding": "#d62728" # Red
}

def load_network(scenario="high_135TWh"):
    fn = os.path.join(RESULTS_DIR, f"network_{scenario}.nc")
    return pypsa.Network(fn)

def plot_generation_mix_bar(comparison_csv):
    """Annual energy mix per scenario"""
    df = pd.read_csv(comparison_csv)
    # Pivot for plotting
    carriers = ['gas_GWh', 'coal_GWh', 'hydro_GWh', 'solar_GWh', 'oil_GWh', 'imports_GWh']
    plot_df = df.set_index('scenario')[carriers]
    # Clean up column names for legend
    plot_df.columns = [c.replace('_GWh', '').capitalize() for c in plot_df.columns]
    
    ax = plot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=[COLORS.get(c.lower(), '#333') for c in plot_df.columns])
    plt.title("Annual Electricity Generation Mix by Scenario (2025)", fontsize=14)
    plt.ylabel("Generation [GWh]")
    plt.xlabel("Demand Scenario")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "generation_mix_comparison.png"), dpi=300)
    print(f"Saved: {PLOTS_DIR}/generation_mix_comparison.png")

def plot_hourly_dispatch(n, scenario_name, period="2025-07-01"):
    """Stacked area chart for a sample week"""
    # Select 1 week
    start = pd.Timestamp(period)
    end = start + pd.Timedelta(days=7)
    
    # Get generation by carrier
    gen = n.generators_t.p.loc[start:end]
    # Group by carrier (Pandas 3.0 compatible)
    gen_carrier = gen.T.groupby(n.generators.carrier).sum().T
    
    # Select winter week
    if "01-07" in period:
        season = "Winter"
    else:
        season = "Summer"
    
    # Add load for reference
    load = n.loads_t.p_set.loc[start:end].sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot stacked area
    gen_carrier.plot.area(ax=ax, color=[COLORS.get(c, '#333') for c in gen_carrier.columns], alpha=0.8)
    
    # Plot load line
    load.plot(ax=ax, color='black', linewidth=2, label='Demand')
    
    plt.title(f"Hourly Dispatch - Uzbekistan ({scenario_name}) - {season} Week", fontsize=14)
    plt.ylabel("Power [MW]")
    plt.xlabel("Date")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"dispatch_{season.lower()}_{scenario_name}.png"), dpi=300)
    print(f"Saved: {PLOTS_DIR}/dispatch_{season.lower()}_{scenario_name}.png")

def plot_cost_breakdown(comparison_csv):
    """Bar chart for total system costs"""
    df = pd.read_csv(comparison_csv)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='scenario', y='total_cost_USD', palette='viridis')
    plt.title("Total Annual System Cost by Scenario (2025)", fontsize=14)
    plt.ylabel("Total Cost [USD]")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "system_costs.png"), dpi=300)
    print(f"Saved: {PLOTS_DIR}/system_costs.png")

if __name__ == "__main__":
    print("Generating visualizations (Berlin Economics Style)...")
    
    # 1. Comparison plots
    comp_file = os.path.join(RESULTS_DIR, "scenario_comparison.csv")
    if os.path.exists(comp_file):
        plot_generation_mix_bar(comp_file)
        plot_cost_breakdown(comp_file)
    
    # 2. Detailed dispatch for High scenario
    try:
        n = load_network("high_135TWh")
        plot_hourly_dispatch(n, "high_135TWh", period="2025-07-07") # Summer peak week
        plot_hourly_dispatch(n, "high_135TWh", period="2025-01-07") # Winter peak week
    except Exception as e:
        print(f"Error loading network for dispatch plot: {e}")

    print("\n🖤 Visualizations complete! Check results/plots/ directory.")
