#!/usr/bin/env python3
"""Core utilities for Berlin Economics replication scenarios.

Scenarios:
- gov_plan_trade
- gov_plan_notrade
- cost_optimal_trade

Each runner script passes:
- capacity_mode: gov_plan | cost_optimal
- trade_enabled: bool

Improvements (v2):
1. Plant bus mapping guardrail — warns & drops unmapped plants
2. Demand schema validation — fails fast on missing bus columns
3. Path portability — --project-dir / PYPSA_UZB_DIR env var
4. Richer run metadata — solver status, gap, runtime in summary CSV
5. Post-solve sanity report — load shedding, import share, new capacity
"""

from __future__ import annotations
import os
import re
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")

# Auto-detect: env var > parent of scripts/ dir > Docker fallback
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTO_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # scripts/ -> project root
DEFAULT_PROJECT_DIR = os.environ.get("PYPSA_UZB_DIR", _AUTO_PROJECT_DIR)

SOLVER = "highs"
# Threads: default to (logical CPUs - 2), leaving 2 cores free for other apps,
# so the solver adapts to whatever machine runs it. Override with the
# SOLVER_THREADS env var, e.g. `SOLVER_THREADS=12 ...` to use all cores.
SOLVER_THREADS = int(os.environ.get("SOLVER_THREADS") or max(1, (os.cpu_count() or 4) - 2))
# Interior-point + crossover handles large extendable-capacity LPs better than simplex.
# Crossover converts the IPM solution to a basic one to avoid small bound violations.
SOLVER_OPTIONS = {
    "time_limit": 86400,        # 24 hours
    "threads": SOLVER_THREADS,
    "solver": "ipm",            # interior point method
    "run_crossover": "on",      # produces basic solution from IPM
    "presolve": "on",
}
DISCOUNT_RATE = 0.07

BUS_INFO = {
    "central":   {"x": 69.3, "y": 41.3, "country": "UZ", "type": "domestic"},
    "east":      {"x": 71.0, "y": 40.5, "country": "UZ", "type": "domestic"},
    "south":     {"x": 69.0, "y": 39.0, "country": "UZ", "type": "domestic"},
    "southwest": {"x": 65.5, "y": 39.0, "country": "UZ", "type": "domestic"},
    "northwest": {"x": 59.5, "y": 42.0, "country": "UZ", "type": "domestic"},
    "kazakhstan":   {"x": 68.0, "y": 43.0, "country": "KZ", "type": "international"},
    "kyrgyzstan":   {"x": 74.5, "y": 41.0, "country": "KG", "type": "international"},
    "tajikistan":   {"x": 69.0, "y": 38.5, "country": "TJ", "type": "international"},
    "turkmenistan": {"x": 58.5, "y": 40.0, "country": "TM", "type": "international"},
}
DOMESTIC_BUSES = [b for b, i in BUS_INFO.items() if i["type"] == "domestic"]
INTL_BUSES = [b for b, i in BUS_INFO.items() if i["type"] == "international"]

PLANT_BUS_MAP = {
    "Angren power station": "east", "Novo-Angren power station": "east",
    "Mubarek": "southwest", "Navoi": "southwest", "Syrdarya": "central",
    "Takhiatash": "northwest", "Talimardjan": "southwest", "Tashkent": "central",
    "Akkavak": "central", "Charvak": "central", "Farkhad": "central",
    "Gazalkent": "central", "Khodjikent": "central", "Tavak": "central",
    "Fergana": "east", "Sazagan": "southwest",
}

MARGINAL_COSTS = {"Gas": 25.0, "Coal": 35.0, "Oil": 55.0, "Hydro": 2.0, "Solar": 0.0, "Wind": 0.0}
EFFICIENCY = {"Gas": 0.40, "Coal": 0.33, "Oil": 0.35, "Hydro": 1.0, "Solar": 1.0, "Wind": 1.0}
AVAILABILITY = {"Gas": 0.95, "Coal": 0.85, "Oil": 0.90}
# Coal min-stable-gen set to 0 (was 0.60) so coal can cycle to zero under high carbon
# pricing — effectively models retirement of Angren / Novo-Angren. Capacity isn't removed
# from the network, but if the optimizer never dispatches a plant, it's economically retired.
MIN_STABLE_GEN = {"Gas": 0.50, "Coal": 0.0, "Oil": 0.30, "Hydro": 0.0, "Solar": 0.0, "Wind": 0.0}
IMPORT_PRICES = {"kazakhstan": 45.0, "kyrgyzstan": 35.0, "tajikistan": 35.0, "turkmenistan": 40.0}

GOV_PLAN_ADDITIONS = {
    "central":   {"Solar": 1000, "Wind": 1000, "Gas": 2500, "BESS": 1000, "PumpedHydro": 500, "Hydro": 500},
    "east":      {"Solar": 1000, "Wind": 1000, "Gas": 1000, "BESS": 500,  "PumpedHydro": 500, "Hydro": 300},
    "south":     {"Solar": 2000, "Wind": 1000, "Gas": 1000, "BESS": 500,  "PumpedHydro": 500, "Hydro": 400},
    "southwest": {"Solar": 2500, "Wind": 9000, "Gas": 1500, "BESS": 500,  "PumpedHydro": 500, "Hydro": 400},
    "northwest": {"Solar": 2500, "Wind": 9000, "Gas": 1000, "BESS": 500,  "PumpedHydro": 0,   "Hydro": 400},
}

LIFETIMES = {"Solar": 25, "Wind": 25, "Gas": 30, "BESS": 15, "PumpedHydro": 50, "Line": 40}
CAPEX = {"Solar": 600, "Wind": 1100, "Gas": 900, "BESS_Power": 150, "BESS_Energy": 150, "PumpedHydro": 1500}

# Transmission expansion CAPEX (NREL/EIA midrange for HVAC overhead) — USD per MW per km
TRANSMISSION_CAPEX_PER_MW_KM = 1500.0


def annualize_cost(capex_per_kw: float, lifetime: int, discount_rate: float = DISCOUNT_RATE) -> float:
    capex_per_mw = capex_per_kw * 1000
    crf = (discount_rate * (1 + discount_rate) ** lifetime) / ((1 + discount_rate) ** lifetime - 1)
    return capex_per_mw * crf


ANNUALIZED = {
    "Solar": annualize_cost(CAPEX["Solar"], LIFETIMES["Solar"]),
    "Wind": annualize_cost(CAPEX["Wind"], LIFETIMES["Wind"]),
    "Gas": annualize_cost(CAPEX["Gas"], LIFETIMES["Gas"]),
    "BESS": annualize_cost(CAPEX["BESS_Power"] + 4 * CAPEX["BESS_Energy"], LIFETIMES["BESS"]),
    "PumpedHydro": annualize_cost(CAPEX["PumpedHydro"], LIFETIMES["PumpedHydro"]),
}

# Annualized line CAPEX is USD/MW/km/year — multiplied by line length downstream.
_line_crf = (DISCOUNT_RATE * (1 + DISCOUNT_RATE) ** LIFETIMES["Line"]) / ((1 + DISCOUNT_RATE) ** LIFETIMES["Line"] - 1)
ANNUALIZED_LINE_PER_MW_KM = TRANSMISSION_CAPEX_PER_MW_KM * _line_crf

# ---------------------------------------------------------------------------
# POLICY FRAMEWORK — Framing 2 (no siting, tiered transmission, carbon pricing)
# ---------------------------------------------------------------------------
# • Cost-optimal mode: solar + wind candidates available in ALL 5 domestic zones.
#   The optimizer picks where to build based on resource profile and economics.
# • Inter-zonal transmission lines are tiered:
#       Tier 1 (1× CAPEX): "strategic corridors" connecting industrial demand
#                          centers and resource-rich peripheries to them.
#       Tier 2 (3× CAPEX): all other inter-zonal lines (long, low-capacity, or
#                          duplicative routes).
# • International lines: fixed at existing capacity (no expansion modeled).
TRANSMISSION_TIER_1_LINES = {
    "central_east",         # wind from Central to Fergana industrial
    "central_southwest",    # backbone between two major demand centers
    "south_southwest",      # South resource → SW industrial
    "northwest_southwest",  # NW wind → SW industrial
}
TRANSMISSION_PENALTY_MULTIPLIER = 3.0   # applied to non-Tier-1 domestic lines

# Carbon emissions per MWh of FUEL energy (thermal MWh, before efficiency).
# Used to compute a marginal-cost adder per generator under --carbon-price.
# Numbers match the n.add("Carrier", ..., co2_emissions=...) in build_and_run.
CO2_EMISSIONS = {
    "gas":   0.20,   # tCO2 per MWh_thermal
    "coal":  0.34,
    "oil":   0.26,
    "hydro": 0.0,
    "solar": 0.0,
    "wind":  0.0,
}


def carbon_adder(carrier: str, efficiency: float, carbon_price: float) -> float:
    """USD/MWh_electric marginal-cost adder for a generator at given carbon price.

    Per MWh_el: fuel needed = 1/efficiency MWh_th → CO2 emitted = co2/efficiency
                cost added  = (co2/efficiency) × carbon_price
    """
    if carbon_price <= 0 or efficiency <= 0:
        return 0.0
    co2_th = CO2_EMISSIONS.get(carrier.lower(), 0.0)
    return co2_th * carbon_price / efficiency


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_profile_from_ninja(kind: str, region: str, snapshots: pd.DatetimeIndex, data_dir: str) -> pd.Series:
    path = os.path.join(data_dir, "pypsa-data", "pv-wind", f"{region}_{kind}.csv")
    if not os.path.exists(path):
        return pd.Series(0.0, index=snapshots)
    # Renewables.ninja reports power output scaled to its 'capacity' param (kW).
    # Divide by it to recover a 0-1 capacity factor (p_max_pu). All current files
    # use capacity=1 (so this is a no-op), but earlier downloads used capacity=2
    # for some zones — this guard keeps the loader correct regardless of the value.
    cap = 1.0
    try:
        with open(path) as _fh:
            head = "".join(next(_fh) for _ in range(3))
        m = re.search(r'"capacity":\s*"?([0-9.]+)"?', head)
        if m and float(m.group(1)) > 0:
            cap = float(m.group(1))
    except Exception:
        cap = 1.0
    src = pd.read_csv(path, comment="#")
    src["time"] = pd.to_datetime(src["time"])
    src = src.set_index("time")
    elec = src["electricity"] / cap
    lookup = {(t.month, t.day, t.hour): v for t, v in elec.items()}
    vals = []
    for t in snapshots:
        key = (t.month, t.day, t.hour)
        vals.append(float(lookup.get(key, np.nan)))
    s = pd.Series(vals, index=snapshots).interpolate().bfill().ffill()
    return s.clip(lower=0.0, upper=1.0)


def create_hydro_profile(snapshots: pd.DatetimeIndex) -> pd.Series:
    monthly_cf = {1:0.20,2:0.18,3:0.25,4:0.40,5:0.55,6:0.60,7:0.50,8:0.40,9:0.30,10:0.25,11:0.22,12:0.20}
    return pd.Series([monthly_cf[m] for m in snapshots.month], index=snapshots)


def load_demand_for_year(year: int, demand_file: str, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "pypsa-data", "demand", demand_file)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Improvement 2: Demand schema validation ---
    missing = [b for b in DOMESTIC_BUSES if b not in df.columns]
    if missing:
        raise ValueError(
            f"Demand file {demand_file} missing required bus columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[df.index.year == year].copy()
    if y.empty:
        raise ValueError(f"No rows for year={year} in {demand_file}")
    full = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    return y.reindex(full, method="ffill")


def load_power_plants(data_dir: str) -> pd.DataFrame:
    plants = pd.read_csv(os.path.join(data_dir, "calliope-data/data/uzb_power_plants.csv"), sep=";")
    plants = plants[["name", "capacity_mw", "primary_fuel"]].copy()
    plants["bus"] = plants["name"].map(PLANT_BUS_MAP)
    plants["carrier"] = plants["primary_fuel"].str.strip()

    # --- Improvement 1: Plant bus mapping guardrail ---
    unmapped = plants[plants["bus"].isna()]
    if len(unmapped) > 0:
        total_mw = unmapped["capacity_mw"].sum()
        print(f"  ⚠ Dropping {len(unmapped)} plants with unknown bus mapping ({total_mw:.0f} MW total):",
              file=sys.stderr)
        for _, row in unmapped.iterrows():
            print(f"    - {row['name']} ({row['carrier']}, {row['capacity_mw']:.0f} MW)", file=sys.stderr)
        plants = plants.dropna(subset=["bus"])

    plants["marginal_cost"] = plants["carrier"].map(MARGINAL_COSTS).fillna(30.0)
    plants["efficiency"] = plants["carrier"].map(EFFICIENCY).fillna(0.40)
    return plants


# ---------------------------------------------------------------------------
# Sanity report (Improvement 5)
# ---------------------------------------------------------------------------

def print_sanity_report(
    n: pypsa.Network,
    scenario_name: str,
    year: int,
    capacity_mode: str,
    trade_enabled: bool,
    solve_seconds: float,
    solver_status: str,
    termination: str,
) -> None:
    """Print a quick post-solve sanity check."""
    total_demand_mwh = n.loads_t.p_set.sum().sum()

    # Load shedding
    shed_gens = [g for g in n.generators.index if g.startswith("loadshed_")]
    shed_mwh = n.generators_t.p[shed_gens].sum().sum() if shed_gens else 0.0
    shed_gwh = shed_mwh / 1e3
    shed_pct = 100 * shed_mwh / total_demand_mwh if total_demand_mwh > 0 else 0.0

    # Import share
    if trade_enabled:
        imp_gens = [g for g in n.generators.index if g.startswith("import_")]
        imp_mwh = n.generators_t.p[imp_gens].sum().sum() if imp_gens else 0.0
        imp_gwh = imp_mwh / 1e3
        imp_pct = 100 * imp_mwh / total_demand_mwh if total_demand_mwh > 0 else 0.0

    # New capacity (cost-optimal only)
    new_cap = {}
    if capacity_mode == "cost_optimal":
        for carrier in ["solar", "wind", "gas"]:
            cands = [g for g in n.generators.index if g.startswith(f"candidate_{carrier}_")]
            if cands:
                new_cap[carrier.capitalize()] = n.generators.loc[cands, "p_nom_opt"].sum()
        for carrier_label, carrier_name in [("BESS", "BESS"), ("PH", "PumpedHydro")]:
            cands = [s for s in n.storage_units.index if f"candidate_{carrier_name.lower()}_" in s.lower()
                     or f"candidate_{carrier_label.lower()}_" in s.lower()]
            if cands:
                new_cap[carrier_label] = n.storage_units.loc[cands, "p_nom_opt"].sum()

    bar = f"═══ Sanity Report: {scenario_name} ({year}) ═══"
    print(f"\n{bar}")
    print(f"  Load shedding:  {shed_gwh:,.1f} GWh ({shed_pct:.2f}% of demand)")
    if trade_enabled:
        print(f"  Import share:   {imp_gwh:,.1f} GWh ({imp_pct:.1f}% of demand)  [cap: 20%]")
    else:
        print(f"  Import share:   disabled (no-trade scenario)")
    if new_cap:
        parts = " | ".join(f"{k} {v:,.0f} MW" for k, v in new_cap.items())
        print(f"  New capacity:   {parts}")

    # Renewable curtailment summary (applies to both cost_optimal and gov_plan)
    curtailed_total = 0.0
    potential_total = 0.0
    curt_by_carrier = {}
    for g in n.generators.index:
        carrier = n.generators.loc[g, "carrier"]
        if carrier not in ("solar", "wind"):
            continue
        if n.generators.loc[g, "bus"] not in n.buses.index:
            continue
        p_nom = (n.generators.loc[g, "p_nom_opt"]
                 if n.generators.loc[g, "p_nom_extendable"]
                 else n.generators.loc[g, "p_nom"])
        if p_nom < 0.1:
            continue
        if g in n.generators_t.p_max_pu.columns:
            potential = (n.generators_t.p_max_pu[g] * p_nom).sum()
        else:
            potential = float(n.generators.loc[g, "p_max_pu"]) * p_nom * len(n.snapshots)
        actual = n.generators_t.p[g].sum() if g in n.generators_t.p.columns else 0.0
        curtailed = max(0.0, potential - actual)
        curtailed_total += curtailed
        potential_total += potential
        curt_by_carrier[carrier] = curt_by_carrier.get(carrier, 0.0) + curtailed
    if potential_total > 0:
        curt_pct = 100 * curtailed_total / potential_total
        parts = " | ".join(f"{k} {v/1e3:,.1f} GWh" for k, v in curt_by_carrier.items() if v > 0.5)
        print(f"  RE curtailment: {curtailed_total/1e3:,.1f} GWh "
              f"({curt_pct:.1f}% of potential)" + (f"  [{parts}]" if parts else ""))

    # Transmission expansion (both cost_optimal and gov_plan now support it)
    if "s_nom_opt" in n.lines.columns:
        ext_lines = n.lines[n.lines.s_nom_extendable].copy()
        if len(ext_lines) > 0:
            ext_lines["new_MW"] = (ext_lines["s_nom_opt"] - ext_lines["s_nom"]).clip(lower=0)
            built = ext_lines[ext_lines["new_MW"] > 0.5]
            if len(built) > 0:
                pieces = " | ".join(f"{idx} +{row.new_MW:,.0f} MW" for idx, row in built.iterrows())
                print(f"  New lines:      {pieces}")
            else:
                print(f"  New lines:      none (no transmission expansion built)")

    print(f"  Solver:         {termination} in {solve_seconds:.1f}s")
    print(f"{'═' * len(bar)}\n")


# ---------------------------------------------------------------------------
# Main build & run
# ---------------------------------------------------------------------------

def build_and_run(
    scenario_name: str,
    capacity_mode: str,
    trade_enabled: bool,
    year: int,
    demand_file: str,
    project_dir: str = DEFAULT_PROJECT_DIR,
    carbon_price: float = 0.0,
) -> tuple[pypsa.Network, dict]:
    data_dir = os.path.join(project_dir, "data")
    results_dir = os.path.join(project_dir, "results", "policy_aligned")
    os.makedirs(results_dir, exist_ok=True)

    # Carbon price (USD/tCO2) — adder applied to gas/coal/oil per-generator using
    # their efficiency. New CCGT: +0.36×CP, existing gas: +0.50×CP, coal: +1.03×CP.
    gas_mc_new = MARGINAL_COSTS["Gas"] + carbon_adder("gas", 0.55, carbon_price)

    n = pypsa.Network(name=f"UZB_{scenario_name}_{year}")
    snapshots = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    n.set_snapshots(snapshots)

    for carrier, co2 in [("gas",0.20),("coal",0.34),("oil",0.26),("hydro",0.0),("solar",0.0),("wind",0.0),("BESS",0.0),("PumpedHydro",0.0),("AC",0.0)]:
        n.add("Carrier", carrier, co2_emissions=co2)

    for bus, info in BUS_INFO.items():
        if info["type"] == "international" and not trade_enabled:
            continue
        n.add("Bus", bus, carrier="AC", x=info["x"], y=info["y"], country=info["country"])

    lines = pd.read_csv(os.path.join(data_dir, "transmission_network.csv"))
    for _, r in lines.iterrows():
        if (r["link_type"] == "international") and (not trade_enabled):
            continue
        if r["bus0"] not in n.buses.index or r["bus1"] not in n.buses.index:
            continue

        s_nom_existing = float(r["nominal_capacity_MW"])
        length_km = float(r["length_km"])
        line_kwargs = dict(
            bus0=r["bus0"], bus1=r["bus1"], s_nom=s_nom_existing,
            length=length_km, x=float(r["reactance_ohm"]),
            r=float(r["resistance_ohm"]), s_max_pu=0.7,
        )

        # Transmission expansion: only enabled in cost_optimal mode for DOMESTIC lines.
        # International + gov_plan: fixed at existing capacity.
        # Transmission expansion enabled in BOTH cost_optimal and gov_plan for domestic
        # inter-zonal lines. Same tiered CAPEX (1× on strategic corridors, 3× on others).
        # Lets gov_plan model resolve the curtailment problem via grid investment.
        # International lines stay fixed in all modes.
        is_domestic = (r["link_type"] == "domestic")
        if is_domestic:
            multiplier = 1.0 if r["link_name"] in TRANSMISSION_TIER_1_LINES else TRANSMISSION_PENALTY_MULTIPLIER
            line_kwargs.update(
                s_nom_extendable=True,
                s_nom_min=s_nom_existing,
                capital_cost=ANNUALIZED_LINE_PER_MW_KM * length_km * multiplier,
            )

        n.add("Line", r["link_name"], **line_kwargs)

    hydro_profile = create_hydro_profile(snapshots)
    solar_profiles = {b: load_profile_from_ninja("pv", b, snapshots, data_dir) for b in DOMESTIC_BUSES}
    wind_profiles = {b: load_profile_from_ninja("wind", b, snapshots, data_dir) for b in DOMESTIC_BUSES}

    plants = load_power_plants(data_dir)
    # Apply carbon-price adder to existing fossil plants (per-plant efficiency).
    if carbon_price > 0:
        plants["marginal_cost"] = plants.apply(
            lambda r: r["marginal_cost"] + carbon_adder(r["carrier"], r["efficiency"], carbon_price),
            axis=1,
        )
    for _, p in plants.iterrows():
        carrier = p["carrier"].lower()
        if carrier == "hydro":
            p_max_pu = hydro_profile
        elif carrier == "solar":
            p_max_pu = solar_profiles.get(p["bus"], pd.Series(0.0, index=snapshots))
        else:
            p_max_pu = AVAILABILITY.get(p["carrier"], 0.95)
        n.add("Generator", p["name"], bus=p["bus"], carrier=carrier, p_nom=float(p["capacity_mw"]),
              marginal_cost=float(p["marginal_cost"]), efficiency=float(p["efficiency"]),
              p_max_pu=p_max_pu, p_min_pu=MIN_STABLE_GEN.get(p["carrier"], 0.0))

    demand = load_demand_for_year(year, demand_file, data_dir)
    for b in DOMESTIC_BUSES:
        if b in demand.columns:
            n.add("Load", f"load_{b}", bus=b, p_set=demand[b])

    if trade_enabled:
        caps = {"tajikistan":5300,"kyrgyzstan":3700,"kazakhstan":2500,"turkmenistan":1000}
        for b in INTL_BUSES:
            n.add("Generator", f"import_{b}", bus=b, carrier="AC", p_nom=caps.get(b,3000), marginal_cost=IMPORT_PRICES.get(b,35.0), p_max_pu=1.0)
            n.add("Generator", f"export_{b}", bus=b, carrier="AC", p_nom=0.6*caps.get(b,3000), marginal_cost=-15.0, p_max_pu=1.0, sign=-1)

    for b in DOMESTIC_BUSES:
        n.add("Generator", f"loadshed_{b}", bus=b, carrier="AC", p_nom=5000, marginal_cost=1000, p_max_pu=1.0)

    if capacity_mode == "gov_plan":
        # Gov plan locations + MW act as a FLOOR; optimizer can build more on top.
        # Hydro stays fixed (site-specific, no generic CAPEX).
        for b in DOMESTIC_BUSES:
            add = GOV_PLAN_ADDITIONS[b]
            if add["Solar"] > 0:
                n.add("Generator", f"gov_solar_{b}", bus=b, carrier="solar",
                      p_nom_extendable=True, p_nom_min=add["Solar"],
                      capital_cost=ANNUALIZED["Solar"], marginal_cost=0.0, p_max_pu=solar_profiles[b])
            if add["Wind"] > 0:
                n.add("Generator", f"gov_wind_{b}", bus=b, carrier="wind",
                      p_nom_extendable=True, p_nom_min=add["Wind"],
                      capital_cost=ANNUALIZED["Wind"], marginal_cost=0.0, p_max_pu=wind_profiles[b])
            if add["Gas"] > 0:
                n.add("Generator", f"gov_gas_{b}", bus=b, carrier="gas",
                      p_nom_extendable=True, p_nom_min=add["Gas"],
                      capital_cost=ANNUALIZED["Gas"], marginal_cost=gas_mc_new,
                      efficiency=0.55, p_max_pu=0.95)
            if add["Hydro"] > 0:
                n.add("Generator", f"gov_hydro_{b}", bus=b, carrier="hydro", p_nom=add["Hydro"],
                      marginal_cost=MARGINAL_COSTS["Hydro"], p_max_pu=hydro_profile)
            if add["BESS"] > 0:
                n.add("StorageUnit", f"gov_bess_{b}", bus=b, carrier="BESS",
                      p_nom_extendable=True, p_nom_min=add["BESS"],
                      capital_cost=ANNUALIZED["BESS"], max_hours=4.0,
                      efficiency_store=0.95, efficiency_dispatch=0.95, cyclic_state_of_charge=True)
            if add["PumpedHydro"] > 0:
                n.add("StorageUnit", f"gov_ph_{b}", bus=b, carrier="PumpedHydro",
                      p_nom_extendable=True, p_nom_min=add["PumpedHydro"],
                      capital_cost=ANNUALIZED["PumpedHydro"], max_hours=8.0,
                      efficiency_store=0.80, efficiency_dispatch=0.85, cyclic_state_of_charge=True)
    else:
        # cost_optimal mode — Framing 2: solar + wind candidates in ALL 5 zones,
        # optimizer picks where to build based on resource profile + economics.
        for b in DOMESTIC_BUSES:
            n.add("Generator", f"candidate_solar_{b}", bus=b, carrier="solar", p_nom_extendable=True,
                  capital_cost=ANNUALIZED["Solar"], marginal_cost=0.0, p_max_pu=solar_profiles[b])
            n.add("Generator", f"candidate_wind_{b}", bus=b, carrier="wind", p_nom_extendable=True,
                  capital_cost=ANNUALIZED["Wind"], marginal_cost=0.0, p_max_pu=wind_profiles[b])
            n.add("Generator", f"candidate_gas_{b}", bus=b, carrier="gas", p_nom_extendable=True,
                  capital_cost=ANNUALIZED["Gas"], marginal_cost=gas_mc_new, efficiency=0.55, p_max_pu=0.95)
            n.add("StorageUnit", f"candidate_bess_{b}", bus=b, carrier="BESS", p_nom_extendable=True,
                  capital_cost=ANNUALIZED["BESS"], max_hours=4.0, efficiency_store=0.95, efficiency_dispatch=0.95,
                  cyclic_state_of_charge=True)
            # PumpedHydro candidate dropped (site-specific; was inflating LP size in cost_optimal).
            # Gov-plan retains PH as fixed/floor allocation per GOV_PLAN_ADDITIONS.

    m = n.optimize.create_model()
    if trade_enabled:
        total_demand = n.loads_t.p_set.sum().sum()
        imp = [g for g in n.generators.index if g.startswith("import_")]
        if imp:
            total_import = m.variables["Generator-p"].loc[:, imp].sum()
            m.add_constraints(total_import <= 0.2 * total_demand, name="import_cap_20pct")
        # Export cap prevents an unbounded direction where the model builds infinite
        # renewables to chase the -$15/MWh export revenue. Cap at 10% of annual demand.
        exp = [g for g in n.generators.index if g.startswith("export_")]
        if exp:
            total_export = m.variables["Generator-p"].loc[:, exp].sum()
            m.add_constraints(total_export <= 0.1 * total_demand, name="export_cap_10pct")

    # --- Improvement 4: Track solve timing ---
    t0 = time.time()
    result = n.optimize.solve_model(solver_name=SOLVER, solver_options=SOLVER_OPTIONS)
    solve_seconds = time.time() - t0

    # Parse solver return (handle tuple or single value)
    if isinstance(result, tuple):
        solver_status, termination = str(result[0]), str(result[1])
    else:
        solver_status, termination = str(result), "unknown"

    if n.objective is None:
        raise RuntimeError(f"Optimization failed: status={solver_status}, termination={termination}")

    # --- Improvement 5: Sanity report ---
    print_sanity_report(n, scenario_name, year, capacity_mode, trade_enabled, solve_seconds, solver_status, termination)

    out_nc = os.path.join(results_dir, f"network_{scenario_name}_{year}.nc")
    n.export_to_netcdf(out_nc)

    gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().T.sum() / 1e3

    # Curtailment (solar + wind, summed across all zones)
    curt_total_mwh = 0.0
    potential_total_mwh = 0.0
    curt_solar_mwh = 0.0
    curt_wind_mwh = 0.0
    for g in n.generators.index:
        carrier = n.generators.loc[g, "carrier"]
        if carrier not in ("solar", "wind"):
            continue
        p_nom = (n.generators.loc[g, "p_nom_opt"]
                 if n.generators.loc[g, "p_nom_extendable"]
                 else n.generators.loc[g, "p_nom"])
        if p_nom < 0.1:
            continue
        if g in n.generators_t.p_max_pu.columns:
            potential = (n.generators_t.p_max_pu[g] * p_nom).sum()
        else:
            potential = float(n.generators.loc[g, "p_max_pu"]) * p_nom * len(n.snapshots)
        actual = n.generators_t.p[g].sum() if g in n.generators_t.p.columns else 0.0
        curt = max(0.0, potential - actual)
        curt_total_mwh += curt
        potential_total_mwh += potential
        if carrier == "solar":
            curt_solar_mwh += curt
        else:
            curt_wind_mwh += curt
    curt_pct = (100 * curt_total_mwh / potential_total_mwh) if potential_total_mwh > 0 else 0.0

    summary = {
        "scenario": scenario_name,
        "year": year,
        "carbon_price_USD_per_tCO2": carbon_price,
        "demand_file": demand_file,
        "total_cost_USD": float(n.objective),
        "gas_GWh": float(gen.get("gas", 0.0)),
        "coal_GWh": float(gen.get("coal", 0.0)),
        "hydro_GWh": float(gen.get("hydro", 0.0)),
        "solar_GWh": float(gen.get("solar", 0.0)),
        "wind_GWh": float(gen.get("wind", 0.0)),
        "imports_GWh": float(n.generators_t.p[[g for g in n.generators.index if g.startswith("import_")]].sum().sum()/1e3) if trade_enabled else 0.0,
        "exports_GWh": float(n.generators_t.p[[g for g in n.generators.index if g.startswith("export_")]].abs().sum().sum()/1e3) if trade_enabled else 0.0,
        # Curtailment metrics
        "re_potential_GWh": round(potential_total_mwh / 1e3, 1),
        "re_curtailed_GWh": round(curt_total_mwh / 1e3, 1),
        "re_curtailed_pct": round(curt_pct, 2),
        "solar_curtailed_GWh": round(curt_solar_mwh / 1e3, 1),
        "wind_curtailed_GWh": round(curt_wind_mwh / 1e3, 1),
        # Solver metadata
        "solver_status": solver_status,
        "termination": termination,
        "solve_seconds": round(solve_seconds, 1),
        "objective_value": float(n.objective),
    }

    # Add optimal capacities for cost-optimal runs
    if capacity_mode == "cost_optimal":
        for carrier in ["solar", "wind", "gas"]:
            cands = [g for g in n.generators.index if g.startswith(f"candidate_{carrier}_")]
            if cands:
                summary[f"new_{carrier}_MW"] = float(n.generators.loc[cands, "p_nom_opt"].sum())
        for label, name in [("bess", "bess"), ("ph", "ph")]:
            cands = [s for s in n.storage_units.index if s.startswith(f"candidate_{name}_")]
            if cands:
                summary[f"new_{label}_MW"] = float(n.storage_units.loc[cands, "p_nom_opt"].sum())

    pd.DataFrame([summary]).to_csv(os.path.join(results_dir, f"summary_{scenario_name}_{year}.csv"), index=False)
    return n, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def common_parser(default_year: int = 2030, default_demand_file: str = ""):
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=default_year, choices=[2030, 2035])
    p.add_argument("--demand-file", type=str, default=default_demand_file)
    # --- Improvement 3: Path portability ---
    p.add_argument("--project-dir", type=str, default=DEFAULT_PROJECT_DIR,
                   help=f"Project root (default: $PYPSA_UZB_DIR or {DEFAULT_PROJECT_DIR})")
    p.add_argument("--carbon-price", type=float, default=0.0,
                   help="Carbon price USD/tCO2 (default: 0). Adds per-MWh cost to gas/coal/oil "
                        "based on their CO2 intensity and efficiency. Suffixes scenario name "
                        "with _co2{N}. Conversion: new CCGT marginal = $25 + 0.364×CP; "
                        "existing gas = $25 + 0.50×CP; coal = $35 + 1.03×CP.")
    return p
