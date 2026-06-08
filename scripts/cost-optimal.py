#!/usr/bin/env python3
from berlin_scenarios_core import build_and_run, common_parser

if __name__ == "__main__":
    parser = common_parser(default_year=2030, default_demand_file="")
    args = parser.parse_args()
    demand_file = args.demand_file or (
        "forecast_scenario1_2030_135TWh_hourly.csv" if args.year == 2030 else "forecast_scenario1_2035_150TWh_hourly.csv"
    )
    suffix = f"_co2{int(args.carbon_price)}" if args.carbon_price > 0 else ""
    _, s = build_and_run(
        scenario_name=f"cost_optimal_trade{suffix}",
        capacity_mode="cost_optimal",
        trade_enabled=True,
        year=args.year,
        demand_file=demand_file,
        project_dir=args.project_dir,
        carbon_price=args.carbon_price,
    )
    print("Done:", s)
