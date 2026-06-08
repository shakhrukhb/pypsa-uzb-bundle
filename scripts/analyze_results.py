#!/usr/bin/env python3
"""Compare converged PyPSA networks across scenarios.

Reads the three .nc files in results/policy_aligned/ and produces:
- Per-zone capacity by carrier (existing + new)
- Gov-plan extension above floor (how much optimizer added)
- Transmission line utilization (avg / max / p99)
- Renewable curtailment per zone
- Marginal prices per zone (load-weighted average + duration curve summary)
- Side-by-side cost / generation / capacity comparison

Writes CSVs to results/policy_aligned/analysis/.
"""
from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
import pypsa
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "policy_aligned")
OUT_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Reuse the model's own cost parameters so the overnight-CAPEX reconstruction
# stays in sync with whatever berlin_scenarios_core.py used.
sys.path.insert(0, SCRIPT_DIR)
from berlin_scenarios_core import LIFETIMES, DISCOUNT_RATE  # noqa: E402

DOMESTIC_ZONES = ["central", "east", "south", "southwest", "northwest"]

# Deemed-energy proxy for the cost of curtailed renewable energy. Under Uzbekistan's
# IPP PPAs (avg ~$0.05/kWh) with take-or-pay / deemed-energy clauses, the offtaker
# pays for energy that is curtailed for grid reasons. We KEEP the gov-owned CAPEX
# optimization framework unchanged; this is a REPORT-ONLY memo cost and does NOT
# enter n.objective. (Households/prosumers sell ~$0.10/kWh; IPP/household share
# unknown, so we use the IPP average as a conservative single rate.)
CURTAILMENT_PPA_USD_PER_MWH = 50.0  # $0.05/kWh

# network carrier (lowercase) -> LIFETIMES key, for recovering overnight CAPEX
# from the annualized capital_cost stored on each component.
_CARRIER_LIFETIME = {
    "solar": "Solar", "wind": "Wind", "gas": "Gas",
    "bess": "BESS", "pumpedhydro": "PumpedHydro",
}


def _crf(lifetime: int, r: float = DISCOUNT_RATE) -> float:
    """Capital recovery factor — inverse converts annualized cost back to overnight."""
    return (r * (1 + r) ** lifetime) / ((1 + r) ** lifetime - 1)


def cost_breakdown(n: pypsa.Network) -> dict:
    """Decompose the total annualized system cost (n.objective) into:

    - operational_USD       : annual OPEX = sum(marginal_cost x weighted dispatch)
                              over all generators (fuel + carbon adder + import
                              purchases - export revenue + load-shed penalty) and
                              any storage with a marginal cost.
    - annualized_capex_USD  : the annualized capital component embedded in the
                              objective. Computed as objective - operational so
                              that operational + annualized_capex == objective
                              exactly (this is the capex PyPSA actually charges,
                              i.e. capital_cost x p_nom_opt of extendable assets).
    - overnight_capex_USD   : one-time, undiscounted investment for NEW capacity
                              only (p_nom_opt - p_nom). Recovered per component as
                              new_MW x capital_cost / CRF(lifetime), so it reflects
                              the same $/kW the model used. A stock ($), not a flow.

    Note the slight asymmetry: annualized_capex reflects everything in the
    objective (including any existing extendable-line capex), whereas
    overnight_capex counts only newly built capacity. Each is faithful to its
    own question (what's in the annual total vs. how much upfront money is needed).
    """
    w = n.snapshot_weightings.objective

    # --- operational ($/yr) ---
    energy = n.generators_t.p.multiply(w, axis=0).sum()  # weighted MWh/yr per gen
    mc_tv = getattr(n.generators_t, "marginal_cost", None)
    if mc_tv is not None and not mc_tv.empty:
        opex = 0.0
        for gen in energy.index:
            if gen in mc_tv.columns:
                opex += (n.generators_t.p[gen] * w * mc_tv[gen]).sum()
            else:
                opex += energy[gen] * n.generators.marginal_cost.get(gen, 0.0)
    else:
        opex = float((energy * n.generators.marginal_cost.reindex(energy.index).fillna(0.0)).sum())

    if len(n.storage_units) and (n.storage_units.marginal_cost.abs() > 0).any():
        su_e = n.storage_units_t.p.clip(lower=0).multiply(w, axis=0).sum()
        opex += float((su_e * n.storage_units.marginal_cost.reindex(su_e.index).fillna(0.0)).sum())

    # --- annualized capex ($/yr): residual so the decomposition reconciles ---
    annualized_capex = float(n.objective) - float(opex)

    # --- overnight capex ($, one-time) for newly built capacity ---
    overnight = 0.0
    g = n.generators
    for idx, row in g[g.p_nom_extendable].iterrows():
        life = _CARRIER_LIFETIME.get(str(row.carrier).lower())
        if life and row.capital_cost > 0:
            new_mw = max(0.0, row.p_nom_opt - row.p_nom)
            overnight += new_mw * row.capital_cost / _crf(LIFETIMES[life])
    if len(n.storage_units):
        su = n.storage_units
        for idx, row in su[su.p_nom_extendable].iterrows():
            life = _CARRIER_LIFETIME.get(str(row.carrier).lower())
            if life and row.capital_cost > 0:
                new_mw = max(0.0, row.p_nom_opt - row.p_nom)
                overnight += new_mw * row.capital_cost / _crf(LIFETIMES[life])
    if "s_nom_opt" in n.lines.columns and n.lines.s_nom_extendable.any():
        for idx, row in n.lines[n.lines.s_nom_extendable].iterrows():
            if row.capital_cost > 0:
                new_mw = max(0.0, row.s_nom_opt - row.s_nom)
                overnight += new_mw * row.capital_cost / _crf(LIFETIMES["Line"])

    return {
        "operational_USD": opex,
        "annualized_capex_USD": annualized_capex,
        "overnight_capex_USD": overnight,
    }


def load_networks() -> dict[str, pypsa.Network]:
    """Auto-discover every network_*.nc in RESULTS_DIR."""
    import glob
    nets = {}
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "network_*.nc")))
    for path in paths:
        fname = os.path.basename(path)
        # strip "network_" prefix and ".nc" suffix; drop trailing year if present
        name = fname[len("network_"):-len(".nc")]
        nets[name] = pypsa.Network(path)
        print(f"  ✓ Loaded {name}")
    if not nets:
        print(f"  ✗ No network_*.nc files in {RESULTS_DIR}", file=sys.stderr)
    return nets


def per_zone_capacity(n: pypsa.Network) -> pd.DataFrame:
    """Capacity by zone × carrier, split into existing/new/total."""
    g = n.generators.copy()
    g = g[g["bus"].isin(DOMESTIC_ZONES)]
    # Existing capacity = plants that are NOT extendable, OR the p_nom_min of extendable gov_* plants
    g["existing_MW"] = np.where(g["p_nom_extendable"], g["p_nom_min"], g["p_nom"])
    g["total_MW"] = g["p_nom_opt"].where(g["p_nom_extendable"], g["p_nom"])
    g["new_MW"] = (g["total_MW"] - g["existing_MW"]).clip(lower=0)

    rows = []
    for (bus, carrier), sub in g.groupby(["bus", "carrier"]):
        rows.append({
            "zone": bus, "carrier": carrier,
            "existing_MW": sub["existing_MW"].sum(),
            "new_MW": sub["new_MW"].sum(),
            "total_MW": sub["total_MW"].sum(),
        })

    # Storage units
    s = n.storage_units.copy()
    s = s[s["bus"].isin(DOMESTIC_ZONES)]
    if len(s):
        s["existing_MW"] = np.where(s["p_nom_extendable"], s["p_nom_min"], s["p_nom"])
        s["total_MW"] = s["p_nom_opt"].where(s["p_nom_extendable"], s["p_nom"])
        s["new_MW"] = (s["total_MW"] - s["existing_MW"]).clip(lower=0)
        for (bus, carrier), sub in s.groupby(["bus", "carrier"]):
            rows.append({
                "zone": bus, "carrier": carrier,
                "existing_MW": sub["existing_MW"].sum(),
                "new_MW": sub["new_MW"].sum(),
                "total_MW": sub["total_MW"].sum(),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["zone", "carrier"]).reset_index(drop=True)
    df = df.round(1)
    return df


def transmission_utilization(n: pypsa.Network) -> pd.DataFrame:
    if "p0" not in n.lines_t:
        return pd.DataFrame()
    flow = n.lines_t.p0
    s_nom = n.lines.s_nom_opt if "s_nom_opt" in n.lines.columns else n.lines.s_nom
    # Use actual installed capacity (s_nom_opt for extendable, s_nom for fixed)
    s_nom = n.lines.apply(lambda r: r["s_nom_opt"] if r["s_nom_extendable"] else r["s_nom"], axis=1)
    rows = []
    for line in flow.columns:
        f = flow[line]
        cap = s_nom[line] if s_nom[line] > 0 else 1.0
        util = f.abs() / cap
        rows.append({
            "line": line,
            "bus0": n.lines.loc[line, "bus0"],
            "bus1": n.lines.loc[line, "bus1"],
            "length_km": round(float(n.lines.loc[line, "length"]), 1),
            "s_nom_MW": n.lines.loc[line, "s_nom"],
            "s_nom_opt_MW": s_nom[line],
            "new_MW": max(0, s_nom[line] - n.lines.loc[line, "s_nom"]),
            "util_avg_%": (util.mean() * 100),
            "util_p99_%": (util.quantile(0.99) * 100),
            "util_max_%": (util.max() * 100),
            "net_flow_GWh": f.sum() / 1e3,
        })
    return pd.DataFrame(rows).round(1).sort_values("util_avg_%", ascending=False).reset_index(drop=True)


def renewable_curtailment(n: pypsa.Network) -> pd.DataFrame:
    """For each renewable generator, % of potential energy curtailed."""
    rows = []
    for g in n.generators.index:
        carrier = n.generators.loc[g, "carrier"]
        bus = n.generators.loc[g, "bus"]
        if carrier not in ("solar", "wind"):
            continue
        if bus not in DOMESTIC_ZONES:
            continue
        if g not in n.generators_t.p.columns:
            continue
        if g not in n.generators_t.p_max_pu.columns:
            # constant p_max_pu — use static value
            p_max_pu = pd.Series(n.generators.loc[g, "p_max_pu"], index=n.snapshots)
        else:
            p_max_pu = n.generators_t.p_max_pu[g]
        p_nom = (n.generators.loc[g, "p_nom_opt"]
                 if n.generators.loc[g, "p_nom_extendable"]
                 else n.generators.loc[g, "p_nom"])
        if p_nom < 0.1:
            continue
        potential = (p_max_pu * p_nom).sum()
        actual = n.generators_t.p[g].sum()
        curtailed = max(0, potential - actual)
        pct = 100 * curtailed / potential if potential > 0 else 0
        rows.append({
            "generator": g, "zone": bus, "carrier": carrier,
            "p_nom_MW": round(p_nom, 1),
            "potential_GWh": round(potential / 1e3, 1),
            "actual_GWh": round(actual / 1e3, 1),
            "curtailed_GWh": round(curtailed / 1e3, 1),
            "curtailment_%": round(pct, 1),
            # Report-only deemed-energy cost (not in objective); see CURTAILMENT_PPA_USD_PER_MWH
            "curtailment_cost_USD": round(curtailed * CURTAILMENT_PPA_USD_PER_MWH, 0),
        })
    return pd.DataFrame(rows).sort_values(["zone", "carrier"]).reset_index(drop=True)


def marginal_prices(n: pypsa.Network) -> pd.DataFrame:
    """Load-weighted average marginal price per zone + duration curve summary."""
    if "marginal_price" not in n.buses_t or n.buses_t.marginal_price.empty:
        return pd.DataFrame()
    mp = n.buses_t.marginal_price
    rows = []
    for z in DOMESTIC_ZONES:
        if z not in mp.columns:
            continue
        price = mp[z]
        load_name = f"load_{z}"
        if load_name in n.loads_t.p_set.columns:
            load = n.loads_t.p_set[load_name]
            load_weighted = (price * load).sum() / load.sum() if load.sum() > 0 else np.nan
        else:
            load_weighted = np.nan
        rows.append({
            "zone": z,
            "mean_USD_per_MWh": round(price.mean(), 2),
            "load_weighted_USD_per_MWh": round(load_weighted, 2),
            "p10_USD_per_MWh": round(price.quantile(0.10), 2),
            "p50_USD_per_MWh": round(price.quantile(0.50), 2),
            "p90_USD_per_MWh": round(price.quantile(0.90), 2),
            "max_USD_per_MWh": round(price.max(), 2),
        })
    return pd.DataFrame(rows)


def scenario_summary(n: pypsa.Network, name: str) -> dict:
    """One-row headline summary."""
    total_demand = n.loads_t.p_set.sum().sum() / 1e3  # GWh
    total_gen = n.generators_t.p.clip(lower=0).sum().sum() / 1e3
    gen_by_carrier = n.generators_t.p.T.groupby(n.generators.carrier).sum().T.sum() / 1e3

    # Capacity totals (domestic only)
    g = n.generators[n.generators["bus"].isin(DOMESTIC_ZONES)].copy()
    g["total_MW"] = np.where(g["p_nom_extendable"], g["p_nom_opt"], g["p_nom"])
    g["existing_MW"] = np.where(g["p_nom_extendable"], g["p_nom_min"], g["p_nom"])
    cap_by_carrier = g.groupby("carrier")[["existing_MW", "total_MW"]].sum()
    cap_by_carrier["new_MW"] = (cap_by_carrier["total_MW"] - cap_by_carrier["existing_MW"]).clip(lower=0)

    # Transmission expansion
    new_line_MW = 0.0
    if "s_nom_opt" in n.lines.columns:
        ext = n.lines[n.lines.s_nom_extendable]
        new_line_MW = (ext["s_nom_opt"] - ext["s_nom"]).clip(lower=0).sum()

    cb = cost_breakdown(n)

    # Report-only curtailment (deemed-energy) cost — NOT added to objective_USD.
    curt_df = renewable_curtailment(n)
    total_curt_GWh = float(curt_df["curtailed_GWh"].sum()) if len(curt_df) else 0.0
    curt_cost_USD = total_curt_GWh * 1e3 * CURTAILMENT_PPA_USD_PER_MWH

    return {
        "scenario": name,
        "objective_USD": round(n.objective, 0),
        "operational_USD": round(cb["operational_USD"], 0),
        "annualized_capex_USD": round(cb["annualized_capex_USD"], 0),
        "overnight_capex_USD": round(cb["overnight_capex_USD"], 0),
        "curtailed_GWh": round(total_curt_GWh, 1),
        "curtailment_cost_USD_at50": round(curt_cost_USD, 0),
        "objective_USD_per_MWh": round(n.objective / (total_demand * 1e3), 2),
        "demand_TWh": round(total_demand / 1e3, 2),
        "total_gen_TWh": round(total_gen / 1e3, 2),
        "new_solar_MW": round(cap_by_carrier.loc["solar", "new_MW"] if "solar" in cap_by_carrier.index else 0, 1),
        "new_wind_MW":  round(cap_by_carrier.loc["wind",  "new_MW"] if "wind"  in cap_by_carrier.index else 0, 1),
        "new_gas_MW":   round(cap_by_carrier.loc["gas",   "new_MW"] if "gas"   in cap_by_carrier.index else 0, 1),
        "new_line_MW":  round(new_line_MW, 1),
        "gas_share_%":    round(100 * gen_by_carrier.get("gas",   0) / total_gen, 1),
        "coal_share_%":   round(100 * gen_by_carrier.get("coal",  0) / total_gen, 1),
        "hydro_share_%":  round(100 * gen_by_carrier.get("hydro", 0) / total_gen, 1),
        "solar_share_%":  round(100 * gen_by_carrier.get("solar", 0) / total_gen, 1),
        "wind_share_%":   round(100 * gen_by_carrier.get("wind",  0) / total_gen, 1),
    }


def main():
    print("Loading networks...")
    nets = load_networks()
    if not nets:
        sys.exit("No networks found.")

    # 1. Scenario summary
    summary_rows = [scenario_summary(n, name) for name, n in nets.items()]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "scenario_summary.csv"), index=False)
    print("\n=== Scenario summary ===")
    print(summary_df.to_string(index=False))

    # 2-5. Per-network detail
    for name, n in nets.items():
        out_subdir = os.path.join(OUT_DIR, name)
        os.makedirs(out_subdir, exist_ok=True)

        cap_df = per_zone_capacity(n)
        cap_df.to_csv(os.path.join(out_subdir, "capacity_by_zone.csv"), index=False)

        tx_df = transmission_utilization(n)
        tx_df.to_csv(os.path.join(out_subdir, "transmission_utilization.csv"), index=False)

        curt_df = renewable_curtailment(n)
        curt_df.to_csv(os.path.join(out_subdir, "curtailment.csv"), index=False)

        price_df = marginal_prices(n)
        price_df.to_csv(os.path.join(out_subdir, "marginal_prices.csv"), index=False)

        print(f"\n=== {name} ===")
        print("\n-- Capacity by zone (MW) --")
        print(cap_df.to_string(index=False))
        print("\n-- Transmission utilization --")
        print(tx_df.to_string(index=False) if len(tx_df) else "(none)")
        print("\n-- Renewable curtailment --")
        print(curt_df.to_string(index=False) if len(curt_df) else "(none)")
        print("\n-- Marginal prices (USD/MWh) --")
        print(price_df.to_string(index=False) if len(price_df) else "(none)")

    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
