# PyPSA Uzbekistan — Setup & Run Guide

Capacity-expansion model of Uzbekistan's 2030 power system. 5-bus zonal
domestic grid + 4 international links (KZ, KG, TJ, TM). Three scenarios:
**cost-optimal**, **gov-plan-trade**, **gov-plan-notrade**, plus gas-price
sensitivity sweep.

---

## 1. Folder structure

```
to_send/
├── data/                                Input data (DO NOT EDIT)
│   ├── pypsa-data/
│   │   ├── pv-wind/                     8760-h Renewables.ninja profiles per zone
│   │   └── demand/                      Hourly demand forecasts (135/130/125 TWh)
│   ├── calliope-data/data/uzb_power_plants.csv     Existing plant fleet
│   ├── transmission_network.csv         5-bus topology + 4 international links
│   └── *.xlsx                           Reference workbooks (not loaded by scripts)
│
├── scripts/                             Active code (single source of truth)
│   ├── berlin_scenarios_core.py         Core: builds network, applies policy, runs HiGHS
│   ├── cost-optimal.py                  Wrapper — cost-minimisation scenario
│   ├── gov-plan-trade.py                Wrapper — government plan with imports/exports
│   ├── gov-plan-notrade.py              Wrapper — government plan, no trade
│   ├── analyze_results.py               Post-processing — reads *.nc files
│   ├── visualize_results.py             Charts from result files
│   └── visualize_grid.py                Network topology map
│
├── results/
│   └── policy_aligned/                  All optimisation outputs land here
│       └── analysis/                    analyze_results.py output (CSVs)
│
├── environment.yml                      Conda env spec (cross-platform)
├── requirements.txt                     pip alternative (Python 3.11)
└── SETUP.md                             This file
```

---

## 2. Environment setup (Linux)

Pick **one** of the two paths.

### Option A — conda (recommended)
```bash
conda env create -f environment.yml
conda activate pypsa-uzb
```

### Option B — pip into a venv
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify install
```bash
python -c "import pypsa, linopy, highspy; print(pypsa.__version__, linopy.__version__, highspy.__version__)"
```
Expected: `1.1.2 0.6.4 1.13.1`

---

## 3. Policy currently encoded (strict γ — Option γ in the modelling log)

- Solar candidate generators only in **east** zone
- Wind candidate generators only in **central** zone
- Gas candidate generators in **all 5** domestic zones (no policy)
- BESS candidate in **all 5** zones (4-hour battery, $750/kW)
- Inter-zonal transmission lines:
  - `central_east` corridor: extendable at **1× CAPEX** ($1500/MW/km, 40-yr lifetime)
  - All other domestic inter-zonal lines: extendable at **3× CAPEX** (policy penalty)
  - International lines: fixed (existing capacity only)
- Import cap: 20% of annual demand
- Export cap: 10% of annual demand (prevents unbounded export-arbitrage)

Government-plan scenarios use fixed plant allocations from
`GOV_PLAN_ADDITIONS` (in `berlin_scenarios_core.py`) as a *floor*, with
`p_nom_extendable=True` so the optimizer can add more on top.

---

## 4. Solver configuration

- Solver: **HiGHS** (open source, ships with `highspy`)
- Method: **interior-point + crossover** (`ipm` + `run_crossover=on`)
- Threads: 8
- Time limit: 86400 s (24 h, generous safety margin — typical solve is ~1 h)

These live in `SOLVER_OPTIONS` near the top of `berlin_scenarios_core.py`
— adjust there if you have more cores or want a shorter time cap.

---

## 5. How to run the three baseline scenarios

```bash
cd to_send

# Scenario 1: cost-optimal (strict γ siting, gas = $25/MWh default)
python scripts/cost-optimal.py 2>&1 | tee results/policy_aligned/run_cost_optimal_trade_2030.log

# Scenario 2: government plan with international trade
python scripts/gov-plan-trade.py 2>&1 | tee results/policy_aligned/run_gov_plan_trade_2030.log

# Scenario 3: government plan, no trade (energy-independent)
python scripts/gov-plan-notrade.py 2>&1 | tee results/policy_aligned/run_gov_plan_notrade_2030.log
```

On Linux (no `caffeinate` needed; just don't suspend the machine).
Each run produces:
- `results/policy_aligned/network_<scenario>_2030.nc`     (full PyPSA network)
- `results/policy_aligned/summary_<scenario>_2030.csv`    (one-line summary)

**Each run takes ~30–70 minutes** on a modern 8-core machine with 16 GB RAM.
Watch the log for `Termination condition: optimal`. If you see
`time_limit` or `infeasible`, see Troubleshooting below.

---

## 6. Gas-price sensitivity sweep

```bash
cd to_send

# Baseline at $25/MWh is the cost-optimal run above. Add 4 more:
for price in 40 60 80 100; do
    python scripts/cost-optimal.py --gas-price $price \
      2>&1 | tee results/policy_aligned/run_cost_optimal_trade_gas${price}_2030.log
done
```

Produces `network_cost_optimal_trade_gas40_2030.nc` … `_gas100_2030.nc`.

`--gas-price` overrides the marginal cost of *all* gas generation (existing
fleet, candidate, and gov-plan additions). Useful for carbon-pricing /
subsidy-reform analysis.

---

## 7. Post-processing

After ANY runs finish:

```bash
python scripts/analyze_results.py
```

Auto-discovers every `network_*.nc` in `results/policy_aligned/` and writes:

- `analysis/scenario_summary.csv`                One row per scenario: cost, generation mix, new capacity
- `analysis/<scenario>/capacity_by_zone.csv`     MW by zone × carrier (existing / new / total)
- `analysis/<scenario>/transmission_utilization.csv`     Avg / p99 / max utilisation per line
- `analysis/<scenario>/curtailment.csv`          Renewable energy curtailed
- `analysis/<scenario>/marginal_prices.csv`      LMPs per zone (mean / load-weighted / quantiles)

---

## 8. Troubleshooting

### Solver hits time limit
- Memory pressure → check RAM usage with `htop` or `free -m`. Solver basis can grow to 1–2 GB.
- Reduce `time_limit` in `SOLVER_OPTIONS`. 24 h is generous; you can drop to 4 h (`14400`) without harming a healthy converge.

### Solver returns "infeasible"
- Demand exceeds total generation capability. Check `LOAD_SHEDDING` in the sanity report at the end of the log.
- If you tightened siting constraints, the model may have no way to serve some zone.

### "ValueError: Empty LHS with non-zero RHS in nodal balance constraint"
- Some zone has a load but no generator (existing or candidate). Common after editing `PLANT_BUS_MAP` or removing a candidate type.

### Different results between machines
- HiGHS is deterministic given the same input + options. Small numerical differences (<0.01%) can come from differing BLAS implementations; structural results (which generators get built, transmission MW) should match exactly.

---

## 9. Key parameters to edit

Everything tunable lives at the top of `scripts/berlin_scenarios_core.py`:

| Constant | Default | Meaning |
|---|---|---|
| `CAPEX["Solar"]` | 600 USD/kW | Solar CAPEX 2030 horizon |
| `CAPEX["Wind"]` | 1100 USD/kW | Wind CAPEX 2030 horizon |
| `CAPEX["Gas"]` | 900 USD/kW | New CCGT CAPEX |
| `CAPEX["BESS_Power"] + 4*CAPEX["BESS_Energy"]` | 750 USD/kW | 4-h battery |
| `MARGINAL_COSTS["Gas"]` | 25 USD/MWh | Domestic gas (overridable via `--gas-price`) |
| `MARGINAL_COSTS["Coal"]` | 35 USD/MWh | Domestic coal |
| `IMPORT_PRICES[...]` | 35–45 USD/MWh | Per-neighbor import price |
| `TRANSMISSION_CAPEX_PER_MW_KM` | 1500 USD/MW/km | New HVAC overhead line CAPEX |
| `TRANSMISSION_PENALTY_MULTIPLIER` | 3.0 | Penalty on all inter-zonal except `central_east` |
| `SOLAR_PRIMARY_ZONE` | "east" | Where solar candidates allowed |
| `WIND_PRIMARY_ZONE` | "central" | Where wind candidates allowed |
| `GOV_PLAN_ADDITIONS` | dict | Government plan MW by zone × carrier (used by gov-plan scenarios) |
