#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Carbon-price sweep — Uzbekistan-calibrated levels.
#
# Carbon prices are set deliberately LOW ($5/10/15/20/30 per tCO2), far below
# EU ETS (~$60-90), because a high carbon tax would be unaffordable for
# Uzbek industry — most generation is subsidised domestic gas and businesses
# would be pushed toward bankruptcy. This sweep probes the realistic local
# policy range. (Conversion: new CCGT marginal = $25 + 0.364 x CP, so even
# $30/tCO2 only adds ~$11/MWh to gas.)
#
# This single file is the source of truth for the canonical carbon sweep.
# It ALSO refreshes the $0 baselines, which is required after the 2026-06-08
# wind capacity-factor fix (all prior results are stale).
#
# Runs locally on the user's machine (NOT auto-executed by Claude — solver
# runs are the user's responsibility). Each scenario ~24-70 min.
#
# Usage:
#   conda activate pypsa-mcp        # or your env-mac
#   bash scripts/run_carbon_sweep.sh
# Optional: PYTHON=/path/to/python bash scripts/run_carbon_sweep.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

PRICES=(5 10 15 20 30)            # USD per tCO2 — Uzbekistan-calibrated

# Pick an interpreter: $PYTHON override > python > python3. Modern macOS ships
# only `python3`, so fall back to it if `python` is absent.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python  >/dev/null 2>&1; then PY=python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else echo "ERROR: no 'python' or 'python3' on PATH. Activate your env (e.g. 'conda activate pypsa-mcp') or set PYTHON=/path/to/python." >&2; exit 1; fi
fi
echo "Using interpreter: $PY ($("$PY" --version 2>&1))"
if ! "$PY" -c "import pypsa" >/dev/null 2>&1; then
  echo "ERROR: '$PY' cannot import pypsa. Install requirements into it, or set PYTHON to the interpreter that has it." >&2
  exit 1
fi

LOGDIR="results/policy_aligned"
mkdir -p "$LOGDIR"

# Keep the Mac awake during long runs if caffeinate exists (no-op on Linux).
CAFF=""
command -v caffeinate >/dev/null 2>&1 && CAFF="caffeinate -i -d -s"

run() {  # $1 = script   $2 = log/scenario id   $3.. = extra args
  local script="$1" id="$2"; shift 2
  echo ">>> $id"
  $CAFF "$PY" "$script" "$@" 2>&1 | tee "$LOGDIR/run_${id}.log"
}

echo "=== Baselines (carbon = \$0) — also refreshes the wind-CF fix ==="
run scripts/cost-optimal.py     cost_optimal_trade_2030
run scripts/gov-plan-trade.py   gov_plan_trade_2030
run scripts/gov-plan-notrade.py gov_plan_notrade_2030

echo "=== Carbon sweep: ${PRICES[*]} USD/tCO2 (cost-optimal + gov-plan-trade) ==="
for p in "${PRICES[@]}"; do
  run scripts/cost-optimal.py   "cost_optimal_trade_co2${p}_2030" --carbon-price "$p"
  run scripts/gov-plan-trade.py "gov_plan_trade_co2${p}_2030"     --carbon-price "$p"
done

echo ""
echo "All runs complete. Post-process with:"
echo "  $PY scripts/analyze_results.py"
