#!/usr/bin/env python3
"""Plot Uzbekistan's 5 model zones + transmission network on a regional map,
highlighting NEW transmission capacity the optimizer builds in a scenario.

- Region polygons: data/pypsa-data/gird-geojson/uzbekistan_regional.geojson
  (14 provinces, grouped here into the 5 model zones and colored).
- Zone nodes: berlin_scenarios_core.BUS_INFO coordinates.
- Lines: results/policy_aligned/analysis/<scenario>/transmission_utilization.csv
  (existing corridors drawn thin/grey; new capacity drawn thick/red + labeled
  with "+MW / km").

Stdlib json + matplotlib only (no geopandas).

Usage:
  python scripts/plot_transmission_map.py [scenario_name]
  # default scenario: gov_plan_trade_2030
"""
from __future__ import annotations
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from berlin_scenarios_core import BUS_INFO, DOMESTIC_BUSES  # noqa: E402

GEOJSON = os.path.join(PROJECT_DIR, "data", "pypsa-data", "gird-geojson", "uzbekistan_regional.geojson")
ANALYSIS_DIR = os.path.join(PROJECT_DIR, "results", "policy_aligned", "analysis")
OUT = os.path.join(PROJECT_DIR, "results", "policy_aligned", "plots", "v2", "08_transmission_map.png")

# Province (ADM1_EN) -> model zone
PROVINCE_ZONE = {
    "Tashkent city": "central", "Tashkent region": "central",
    "Syrdarya region": "central", "Jizzakh region": "central",
    "Namangan region": "east", "Andijan region": "east", "Fergana region": "east",
    "Kashkadarya province": "south", "Surkhandarya region": "south", "Samarkand region": "south",
    "Navoi region": "southwest", "Bukhara region": "southwest",
    "Khorezm region": "northwest", "Republic of Karakalpakstan": "northwest",
}
ZONE_COLOR = {
    "central": "#9ecae1", "east": "#a1d99b", "south": "#fdae6b",
    "southwest": "#fee391", "northwest": "#bcbddc",
}
ZONE_DISTRICT = {  # representative wind-profile district (for labels)
    "central": "Bekabad", "east": "Pop", "south": "Boysun",
    "southwest": "Zarafshan/Tamdy", "northwest": "Kungrad",
}


def iter_rings(geom):
    """Yield exterior rings [(lon,lat),...] from Polygon/MultiPolygon/GeometryCollection."""
    t = geom.get("type")
    if t == "Polygon":
        yield geom["coordinates"][0]
    elif t == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly[0]
    elif t == "GeometryCollection":
        for g in geom.get("geometries", []):
            yield from iter_rings(g)


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "gov_plan_trade_2030"

    data = json.load(open(GEOJSON))
    fig, ax = plt.subplots(figsize=(11, 8))

    # --- regions, colored by model zone ---
    drawn_zones = set()
    for feat in data["features"]:
        name = feat["properties"].get("ADM1_EN", "")
        zone = PROVINCE_ZONE.get(name)
        color = ZONE_COLOR.get(zone, "#eeeeee")
        for ring in iter_rings(feat["geometry"]):
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            lbl = zone if zone and zone not in drawn_zones else None
            ax.fill(xs, ys, color=color, alpha=0.55, zorder=1,
                    label=f"{zone} ({ZONE_DISTRICT.get(zone,'')})" if lbl else None)
            ax.plot(xs, ys, color="white", linewidth=0.8, zorder=2)
            if lbl:
                drawn_zones.add(zone)

    # --- transmission lines from the scenario ---
    tx_csv = os.path.join(ANALYSIS_DIR, scenario, "transmission_utilization.csv")
    new_lines = []
    if os.path.exists(tx_csv):
        tx = pd.read_csv(tx_csv)
        for _, r in tx.iterrows():
            b0, b1 = r["bus0"], r["bus1"]
            if b0 not in BUS_INFO or b1 not in BUS_INFO:
                continue
            if b0 not in DOMESTIC_BUSES or b1 not in DOMESTIC_BUSES:
                continue  # focus on domestic corridors
            x0, y0 = BUS_INFO[b0]["x"], BUS_INFO[b0]["y"]
            x1, y1 = BUS_INFO[b1]["x"], BUS_INFO[b1]["y"]
            new_mw = float(r.get("new_MW", 0) or 0)
            # existing corridor (grey base)
            ax.plot([x0, x1], [y0, y1], color="#555555", linewidth=1.3,
                    zorder=3, solid_capstyle="round")
            if new_mw > 0.1:  # NEW capacity overlay
                ax.plot([x0, x1], [y0, y1], color="#d62728",
                        linewidth=2 + new_mw / 150.0, alpha=0.85, zorder=4,
                        solid_capstyle="round")
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                length = float(r.get("length_km", 0) or 0)
                ax.annotate(f"+{new_mw:.0f} MW\n{length:.0f} km", (mx, my),
                            fontsize=8, color="#d62728", fontweight="bold",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.9),
                            zorder=6)
                new_lines.append((b0, b1, new_mw, length))

    # --- zone nodes + labels ---
    for b in DOMESTIC_BUSES:
        x, y = BUS_INFO[b]["x"], BUS_INFO[b]["y"]
        ax.scatter([x], [y], s=90, color="#222222", zorder=5, edgecolor="white")
        ax.annotate(f"{b.upper()}\n({ZONE_DISTRICT.get(b,'')})", (x, y),
                    textcoords="offset points", xytext=(6, 6), fontsize=9,
                    fontweight="bold", zorder=6)

    ax.set_aspect(1.0 / 0.75)  # rough lat correction (~41°N)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    title = f"Uzbekistan — model zones & transmission ({scenario})"
    if not new_lines:
        title += "\n(no new lines built in this scenario)"
    ax.set_title(title, fontsize=12, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="#555555", lw=1.3, label="existing corridor"),
                Line2D([0], [0], color="#d62728", lw=3, label="NEW capacity built")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.3)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"  wrote {OUT}")
    if new_lines:
        print("  new lines:")
        for b0, b1, mw, km in new_lines:
            print(f"    {b0} <-> {b1}: +{mw:.0f} MW, {km:.0f} km")
    else:
        print(f"  (scenario {scenario} built no new domestic lines)")


if __name__ == "__main__":
    main()
