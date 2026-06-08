#!/usr/bin/env python3
"""Publication-quality plots for the v2 (Framing 2 + coal retirement) results.

Reads from results/policy_aligned/analysis/scenario_summary.csv and the
per-scenario CSVs. Writes PNGs to results/policy_aligned/plots/v2/.

Run as: python scripts/plot_results.py
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "policy_aligned")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots", "v2")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

CARRIER_COLORS = {
    "gas":     "#7a7a7a",
    "coal":    "#1a1a1a",
    "hydro":   "#3690c0",
    "solar":   "#fcb315",
    "wind":    "#74c476",
    "imports": "#c994c7",
    "oil":     "#e6550d",
    "other":   "#bdbdbd",
}

CARBON_PRICES = [0, 5, 10, 15, 20, 30]   # Uzbekistan-calibrated sweep


# ============================================================
# Data loading helpers
# ============================================================

def load_summary() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ANALYSIS_DIR, "scenario_summary.csv"))
    # Strip v1 / strict_gamma / carveout variants — keep only v2 (current default-name)
    keep = df[~df["scenario"].str.contains("_v1|_strict_gamma|_carveout|_gov_v1", regex=True)]
    keep = keep.reset_index(drop=True)
    # Merge curtailment from per-scenario summary CSVs (analyze_results.py doesn't include them)
    curt_cols = ["re_curtailed_pct", "re_curtailed_GWh", "solar_curtailed_GWh", "wind_curtailed_GWh"]
    for col in curt_cols:
        keep[col] = 0.0
    for i, r in keep.iterrows():
        scen = r["scenario"]
        # scenario_summary uses "<scenario>_2030" (with year); per-scenario CSV is "summary_<scenario>_2030.csv"
        ps_path = os.path.join(RESULTS_DIR, f"summary_{scen}.csv")
        if os.path.exists(ps_path):
            ps = pd.read_csv(ps_path).iloc[0]
            for col in curt_cols:
                if col in ps:
                    keep.at[i, col] = float(ps[col])
    return keep


def co2_from_scenario(name: str) -> int:
    """Extract carbon price from scenario name (returns 0 if no _co2N suffix)."""
    if "_co2" not in name:
        return 0
    after = name.split("_co2", 1)[1]
    digits = ""
    for ch in after:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else 0


def cost_opt_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Return cost-optimal scenarios indexed by carbon price."""
    rows = []
    for _, r in df.iterrows():
        if not r["scenario"].startswith("cost_optimal_trade"):
            continue
        cp = co2_from_scenario(r["scenario"])
        if cp not in CARBON_PRICES:
            continue
        rows.append({**r.to_dict(), "carbon_price": cp})
    return pd.DataFrame(rows).sort_values("carbon_price").reset_index(drop=True)


def gov_plan_points(df: pd.DataFrame) -> dict:
    """Map: scenario_label -> (carbon_price, row_dict)."""
    out = {}
    for _, r in df.iterrows():
        s = r["scenario"]
        if not s.startswith("gov_plan"):
            continue
        cp = co2_from_scenario(s)
        label = s.replace(f"_co2{cp}", "").replace("_2030", "")
        out[(label, cp)] = r.to_dict()
    return out


def gov_plan_trade_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Return gov-plan-trade scenarios indexed by carbon price."""
    rows = []
    for _, r in df.iterrows():
        s = r["scenario"]
        if not s.startswith("gov_plan_trade"):
            continue
        cp = co2_from_scenario(s)
        if cp not in CARBON_PRICES:
            continue
        rows.append({**r.to_dict(), "carbon_price": cp})
    return pd.DataFrame(rows).sort_values("carbon_price").reset_index(drop=True)


# ============================================================
# Plot 1 — Cost vs carbon price
# ============================================================

def plot_cost_vs_carbon(df: pd.DataFrame):
    co = cost_opt_sweep(df)
    gov = gov_plan_trade_sweep(df)
    gov_other = gov_plan_points(df)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(co["carbon_price"], co["objective_USD"] / 1e9,
            "o-", lw=2.5, ms=8, color="#2c7fb8", label="Cost-optimal (Framing 2)")
    ax.plot(gov["carbon_price"], gov["objective_USD"] / 1e9,
            "s-", lw=2.5, ms=8, color="#d95f0e", label="Government plan (trade)")

    # Gov-plan-notrade as separate point
    if ("gov_plan_notrade", 0) in gov_other:
        r = gov_other[("gov_plan_notrade", 0)]
        ax.plot(0, r["objective_USD"] / 1e9, marker="D", ms=12, color="#993404",
                label="Gov-plan-notrade, CO₂ $0", linestyle="none",
                markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Carbon price (USD / tCO₂)")
    ax.set_ylabel("Annual system cost (USD billion / yr)")
    ax.set_title("Annual System Cost vs Carbon Price — Uzbekistan 2030 (135 TWh)")
    ax.set_xticks(CARBON_PRICES)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_ylim(bottom=0)

    # Annotate cost-opt + gov values
    for _, r in co.iterrows():
        ax.annotate(f"${r['objective_USD']/1e9:.2f}B",
                    (r["carbon_price"], r["objective_USD"] / 1e9),
                    textcoords="offset points", xytext=(0, -15), ha="center",
                    fontsize=8, color="#2c7fb8")
    for _, r in gov.iterrows():
        ax.annotate(f"${r['objective_USD']/1e9:.2f}B",
                    (r["carbon_price"], r["objective_USD"] / 1e9),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8, color="#d95f0e")

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "01_cost_vs_carbon.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ============================================================
# Plot 2 — Renewable share vs carbon price
# ============================================================

def plot_re_share_vs_carbon(df: pd.DataFrame):
    co = cost_opt_sweep(df)
    co["re_pct"] = co["solar_share_%"] + co["wind_share_%"]
    gov = gov_plan_trade_sweep(df)
    gov["re_pct"] = gov["solar_share_%"] + gov["wind_share_%"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(co["carbon_price"], co["re_pct"], "o-",
            lw=2.5, ms=8, color="#238b45", label="Cost-optimal: Solar + Wind")
    ax.plot(gov["carbon_price"], gov["re_pct"], "s-",
            lw=2.5, ms=8, color="#d95f0e", label="Gov-plan-trade: Solar + Wind")
    ax.plot(co["carbon_price"], co["solar_share_%"], "v--",
            lw=1.3, ms=5, color="#fcb315", label="Cost-opt solar", alpha=0.75)
    ax.plot(co["carbon_price"], co["wind_share_%"], "^--",
            lw=1.3, ms=5, color="#74c476", label="Cost-opt wind", alpha=0.75)

    ax.set_xlabel("Carbon price (USD / tCO₂)")
    ax.set_ylabel("Renewable share of generation (%)")
    ax.set_title("Renewable Energy Share vs Carbon Price")
    ax.set_xticks(CARBON_PRICES)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, max(55, co["re_pct"].max() + 5))
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, "02_re_share_vs_carbon.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ============================================================
# Plot 3 — Generation mix stacked bars
# ============================================================

def plot_generation_mix(df: pd.DataFrame):
    co = cost_opt_sweep(df)
    gov = gov_plan_points(df)

    scenarios = []
    for _, r in co.iterrows():
        scenarios.append((f"Cost-opt\nCO₂ ${int(r['carbon_price'])}", r))
    for key, label in [
        (("gov_plan_trade", 0), "Gov-trade\nCO₂ $0"),
        (("gov_plan_notrade", 0), "Gov-notrade\nCO₂ $0"),
        (("gov_plan_trade", 30), "Gov-trade\nCO₂ $30"),
    ]:
        if key in gov:
            scenarios.append((label, gov[key]))

    labels = [s[0] for s in scenarios]
    # Approximate imports share = 100 - (gas+coal+hydro+solar+wind)
    carriers = ["gas", "coal", "hydro", "solar", "wind"]
    data = {c: [s[1].get(f"{c}_share_%", 0) for s in scenarios] for c in carriers}
    other = [max(0, 100 - sum(data[c][i] for c in carriers)) for i in range(len(scenarios))]
    data["imports/other"] = other

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    plot_order = ["gas", "coal", "hydro", "solar", "wind", "imports/other"]
    colors = [CARRIER_COLORS.get(c, "#bdbdbd") if c != "imports/other"
              else CARRIER_COLORS["imports"] for c in plot_order]
    for carrier, color in zip(plot_order, colors):
        ax.bar(x, data[carrier], bottom=bottom, color=color, label=carrier.capitalize(),
               edgecolor="white", linewidth=0.5)
        bottom = bottom + np.array(data[carrier])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of generation (%)")
    ax.set_title("Generation Mix by Scenario — Uzbekistan 2030 (135 TWh)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6, frameon=False)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "03_generation_mix.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ============================================================
# Plot 4 — New capacity stacked bars (cost-opt sweep)
# ============================================================

def plot_new_capacity(df: pd.DataFrame):
    co = cost_opt_sweep(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(co))
    labels = [f"CO₂\n${cp}" for cp in co["carbon_price"]]

    bottom = np.zeros(len(co))
    for col, color, lab in [
        ("new_solar_MW", CARRIER_COLORS["solar"], "Solar"),
        ("new_wind_MW", CARRIER_COLORS["wind"], "Wind"),
        ("new_gas_MW", CARRIER_COLORS["gas"], "Gas"),
    ]:
        vals = co[col].values / 1000  # MW → GW
        ax.bar(x, vals, bottom=bottom, color=color, label=lab,
               edgecolor="white", linewidth=0.5)
        bottom = bottom + vals

    # New transmission as a separate annotation on top
    for i, v in enumerate(co["new_line_MW"].values):
        if v > 0.5:
            ax.annotate(f"+{int(v)} MW lines",
                        (i, bottom[i] + 0.5), ha="center", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("New capacity (GW)")
    ax.set_title("Cost-Optimal New Capacity by Carbon Price (cumulative by 2030)")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "04_new_capacity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ============================================================
# Plot 5 — Renewable curtailment comparison
# ============================================================

def plot_curtailment(df: pd.DataFrame):
    co = cost_opt_sweep(df)
    gov = gov_plan_points(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: percent curtailment
    bars = []
    for _, r in co.iterrows():
        bars.append((f"CO₂ ${int(r['carbon_price'])}", r["re_curtailed_pct"], "#2c7fb8"))
    for key, label, color in [
        (("gov_plan_trade", 0), "Gov-trade $0", "#d95f0e"),
        (("gov_plan_notrade", 0), "Gov-notrade $0", "#993404"),
        (("gov_plan_trade", 30), "Gov-trade $30", "#cc4c02"),
    ]:
        if key in gov:
            bars.append((label, gov[key]["re_curtailed_pct"], color))

    labels = [b[0] for b in bars]
    pcts = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    x = np.arange(len(bars))
    rects = ax1.bar(x, pcts, color=cols, edgecolor="white", linewidth=0.5)
    for rect, p in zip(rects, pcts):
        h = rect.get_height()
        ax1.annotate(f"{p:.1f}%", (rect.get_x() + rect.get_width()/2, h + 0.5),
                     ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Curtailment (% of RE potential)")
    ax1.set_title("Renewable Curtailment by Scenario")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_ylim(0, max(pcts) * 1.2 + 1)

    # Right: absolute GWh wasted
    bars2 = []
    for _, r in co.iterrows():
        bars2.append((f"CO₂ ${int(r['carbon_price'])}", r["re_curtailed_GWh"], "#2c7fb8"))
    for key, label, color in [
        (("gov_plan_trade", 0), "Gov-trade $0", "#d95f0e"),
        (("gov_plan_notrade", 0), "Gov-notrade $0", "#993404"),
        (("gov_plan_trade", 30), "Gov-trade $30", "#cc4c02"),
    ]:
        if key in gov:
            bars2.append((label, gov[key]["re_curtailed_GWh"], color))

    labels = [b[0] for b in bars2]
    gwh = [b[1] for b in bars2]
    cols = [b[2] for b in bars2]
    x = np.arange(len(bars2))
    rects = ax2.bar(x, gwh, color=cols, edgecolor="white", linewidth=0.5)
    for rect, g in zip(rects, gwh):
        h = rect.get_height()
        if g > 100:
            ax2.annotate(f"{g:,.0f}", (rect.get_x() + rect.get_width()/2, h + max(gwh)*0.01),
                         ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Energy wasted (GWh / yr)")
    ax2.set_title("Absolute Energy Curtailed")

    fig.suptitle("Renewable Curtailment — Cost-Optimal vs Government Plan", y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "05_curtailment.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ============================================================
# Plot 6 — Capacity by zone (cost-opt at CO2 $0 / $15 / $30)
# ============================================================

def plot_capacity_by_zone(df: pd.DataFrame):
    # Show side-by-side: cost-opt CO2 $0 vs CO2 $15 vs CO2 $30
    targets = [
        ("cost_optimal_trade_2030", "CO₂ $0"),
        ("cost_optimal_trade_co215_2030", "CO₂ $15"),
        ("cost_optimal_trade_co230_2030", "CO₂ $30"),
    ]
    zones = ["central", "east", "south", "southwest", "northwest"]
    # Explicit labels — truncating to 5 chars collided southwest->"SOUTH" and
    # northwest->"NORTH". Keep all five distinct.
    zone_labels = {"central": "CENTRAL", "east": "EAST", "south": "SOUTH",
                   "southwest": "SW", "northwest": "NW"}
    carriers_show = ["solar", "wind", "gas"]

    fig, axes = plt.subplots(1, len(targets), figsize=(15, 5), sharey=True)
    for ax, (scen, label) in zip(axes, targets):
        cap_csv = os.path.join(ANALYSIS_DIR, scen, "capacity_by_zone.csv")
        if not os.path.exists(cap_csv):
            ax.text(0.5, 0.5, "(missing)", ha="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        cap = pd.read_csv(cap_csv)
        # Use new_MW for the comparison plot
        x = np.arange(len(zones))
        bottom = np.zeros(len(zones))
        for c in carriers_show:
            vals = []
            for z in zones:
                row = cap[(cap["zone"] == z) & (cap["carrier"] == c)]
                vals.append(row["new_MW"].sum() / 1000 if len(row) else 0)
            ax.bar(x, vals, bottom=bottom, color=CARRIER_COLORS[c],
                   label=c.capitalize(), edgecolor="white", linewidth=0.5)
            bottom = bottom + np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([zone_labels[z] for z in zones])
        ax.set_title(f"Cost-opt {label}")
        if ax is axes[0]:
            ax.set_ylabel("New capacity (GW)")
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Where the model builds: new capacity by zone", y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "06_capacity_by_zone.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out}")


# ============================================================
# Main
# ============================================================

def main():
    if not os.path.exists(os.path.join(ANALYSIS_DIR, "scenario_summary.csv")):
        sys.exit(f"Missing {ANALYSIS_DIR}/scenario_summary.csv — run analyze_results.py first")
    df = load_summary()
    if df.empty:
        sys.exit("No v2 scenarios found in scenario_summary.csv")
    print(f"Writing plots to {PLOTS_DIR}")
    plot_cost_vs_carbon(df)
    plot_re_share_vs_carbon(df)
    plot_generation_mix(df)
    plot_new_capacity(df)
    plot_curtailment(df)
    plot_capacity_by_zone(df)
    # plot_v1_vs_v2 dropped — the _v1 scenarios were archived (pre-120m experiments).
    print("Done.")


if __name__ == "__main__":
    main()
