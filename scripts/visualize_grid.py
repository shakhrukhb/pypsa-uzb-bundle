"""
Quick static map of existing and planned transmission lines.

Usage (from repo root):
    python scripts/curated/visualize_grid.py

Outputs:
    scripts/curated/grid_map.png
    scripts/curated/grid_map.svg

Notes:
    - Uses only stdlib + matplotlib (no geopandas needed).
    - Reads the geojsons you placed in data/curated/gird-geojson/.
    - If shapefiles are present and the optional 'shapefile' package is
      installed, they will also be drawn; otherwise geojson alone is used.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Resolve project root: scripts/curated/visualize_grid.py → scripts → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA_DIR = PROJECT_ROOT / "data" / "curated"
GRID_DIR = CURATED_DATA_DIR / "gird-geojson"  # folder name as provided

EXISTING = {
    "geojson": GRID_DIR / "existingtransmissionlines_0" / "existingtransmissionlines_0.geojson",
    "shp": GRID_DIR / "existingtransmissionlines_0" / "Existing_transmission_lines.shp",
    "dbf": GRID_DIR / "existingtransmissionlines_0" / "Existing_transmission_lines.dbf",
}
FUTURE = {
    "geojson": GRID_DIR / "futuretransmissionlines_0" / "futuretransmissionlines_0.geojson",
    "shp": GRID_DIR / "futuretransmissionlines_0" / "Future_transmission_lines.shp",
    "dbf": GRID_DIR / "futuretransmissionlines_0" / "Future_transmission_lines.dbf",
}
REGION_GEOJSON = GRID_DIR / "uzbekistan_regional.geojson"

OUTPUT_DIR = Path(__file__).resolve().parent / "grid-visual"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_PNG = OUTPUT_DIR / "grid_map.png"
OUTPUT_SVG = OUTPUT_DIR / "grid_map.svg"
OUTPUT_HTML = OUTPUT_DIR / "grid_map.html"


def load_lines_geojson(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "LineString":
            lines.append((coords, feat.get("properties", {}).get("Legend", "")))
        elif geom.get("type") == "MultiLineString":
            for seg in coords:
                lines.append((seg, feat.get("properties", {}).get("Legend", "")))
    return lines


def parse_dbf(dbf_path: Path):
    """
    Minimal DBF parser to extract records as dicts. Supports common field types (C, N).
    """
    with dbf_path.open("rb") as f:
        header = f.read(32)
        num_records = struct.unpack("<I", header[4:8])[0]
        header_len = struct.unpack("<H", header[8:10])[0]
        record_len = struct.unpack("<H", header[10:12])[0]

        # Field descriptors (32 bytes each) until 0x0D
        fields = []
        while True:
            desc = f.read(32)
            if desc[0] == 0x0D:
                break
            name = desc[0:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
            ftype = chr(desc[11])
            flen = desc[16]
            fields.append((name, ftype, flen))

        records = []
        for _ in range(num_records):
            rec_bytes = f.read(record_len)
            if not rec_bytes:
                break
            if rec_bytes[0:1] == b"*":  # deleted
                continue
            pos = 1
            rec = {}
            for name, ftype, flen in fields:
                raw = rec_bytes[pos : pos + flen]
                pos += flen
                txt = raw.decode("ascii", errors="ignore").strip()
                rec[name] = txt
            records.append(rec)
        return records


def parse_shp_lines(shp_path: Path):
    """
    Minimal parser for polyline shapefiles (type 3/13), returns list of coordinate lists.
    """
    lines = []
    with shp_path.open("rb") as f:
        header = f.read(100)
        if len(header) < 100:
            return lines
        # file length words = header[24:28] big endian, not used here
        while True:
            rec_header = f.read(8)
            if len(rec_header) < 8:
                break
            # record number, content length (big endian 16-bit words)
            content_len_words = struct.unpack(">i", rec_header[4:8])[0]
            content_bytes = f.read(content_len_words * 2)
            if len(content_bytes) < 4:
                break
            shape_type = struct.unpack("<i", content_bytes[0:4])[0]
            # only handle polyline / polylineZ
            if shape_type not in (3, 13):
                continue
            # bbox 4*8, numParts int, numPoints int
            offset = 4 + 4 * 8
            num_parts = struct.unpack("<i", content_bytes[offset : offset + 4])[0]
            num_points = struct.unpack("<i", content_bytes[offset + 4 : offset + 8])[0]
            offset += 8
            parts = struct.unpack("<" + "i" * num_parts, content_bytes[offset : offset + 4 * num_parts])
            offset += 4 * num_parts
            points = []
            for i in range(num_points):
                x, y = struct.unpack("<dd", content_bytes[offset + i * 16 : offset + (i + 1) * 16])
                points.append((x, y))
            # split into segments by parts
            for idx, start in enumerate(parts):
                end = parts[idx + 1] if idx + 1 < len(parts) else len(points)
                lines.append(points[start:end])
    return lines


def load_lines_with_voltage(shp_path: Path, dbf_path: Path):
    """
    Pair shapefile geometry with DBF 'Legend' field for voltage.
    """
    if not shp_path.exists() or not dbf_path.exists():
        return []
    records = parse_dbf(dbf_path)
    geoms = parse_shp_lines(shp_path)
    lines = []
    for geom, rec in zip(geoms, records):
        legend = rec.get("Legend", "")
        lines.append((geom, legend))
    return lines


def load_polygons(path: Path):
    """
    Load polygons from a GeoJSON (Polygon or MultiPolygon). Returns list of list-of-rings.
    Each ring is a list of (x, y).
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    polys = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        def add_geom(gtype_inner, coords_inner):
            if gtype_inner == "Polygon":
                polys.append(coords_inner)
            elif gtype_inner == "MultiPolygon":
                for poly in coords_inner:
                    polys.append(poly)

        if gtype == "GeometryCollection":
            for sub in geom.get("geometries", []):
                add_geom(sub.get("type"), sub.get("coordinates", []))
        else:
            add_geom(gtype, coords)
    return polys


def normalize_regions_geojson(data):
    """
    Normalize region GeoJSON to a FeatureCollection of Polygon/MultiPolygon features.
    Leaflet does not reliably render GeometryCollection, so we flatten them here.
    """
    out = {"type": "FeatureCollection", "features": []}
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        props = feat.get("properties", {})

        if gtype == "GeometryCollection":
            for sub in geom.get("geometries", []):
                if sub.get("type") in ("Polygon", "MultiPolygon"):
                    out["features"].append(
                        {"type": "Feature", "properties": props, "geometry": sub}
                    )
        elif gtype in ("Polygon", "MultiPolygon"):
            out["features"].append(feat)
    return out


def convex_hull(points):
    """
    Monotone chain convex hull. Points: list of (x,y). Returns list hull vertices.
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def plot_lines(ax, lines, color, label, linewidth=0.1):
    count = 0
    for seg, _legend in lines:
        if len(seg) < 2:
            continue
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.8)
        count += 1
    ax.plot([], [], color=color, linewidth=linewidth + 1, label=f"{label} (n={count})")


def main():
    existing = load_lines_geojson(EXISTING["geojson"])
    future = load_lines_geojson(FUTURE["geojson"])

    # Note: Shapefiles are in Miller Cylindrical projection (meters) while
    # GeoJSON files and the regional boundary are in WGS84 (lat/lon).
    # We prefer GeoJSON to avoid coordinate mismatch with the background map.
    # The GeoJSON files also contain the 220/500 kV Legend field.
    regions = load_polygons(REGION_GEOJSON)

    if not existing and not future:
        print("No lines found. Check that geojson/shapefile files are present.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw regions first (light background)
    for poly in regions:
        for ring in poly:
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            ax.fill(xs, ys, color="#f5f5f5", alpha=0.7, zorder=0)
            ax.plot(xs, ys, color="#bdbdbd", linewidth=0.2, alpha=0.9, zorder=1)
    # Separate voltages if available
    def split_by_voltage(lines):
        hv = []
        lv = []
        for seg, legend in lines:
            if "500" in str(legend):
                hv.append((seg, legend))
            else:
                lv.append((seg, legend))
        return hv, lv

    # Colors: Blue for existing, Red for planned
    # 500 kV lines are bolder (thicker) than 220 kV
    if existing:
        hv, lv = split_by_voltage(existing)
        if hv:
            plot_lines(ax, hv, color="#0066cc", label="Existing 500 kV", linewidth=0.8)
        if lv:
            plot_lines(ax, lv, color="#3399ff", label="Existing 220 kV", linewidth=0.4)
    if future:
        hv, lv = split_by_voltage(future)
        if hv:
            plot_lines(ax, hv, color="#cc0000", label="Planned 500 kV", linewidth=0.8)
        if lv:
            plot_lines(ax, lv, color="#ff6666", label="Planned 220 kV", linewidth=0.4)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_title("Uzbekistan Transmission: Existing vs Planned")
    ax.grid(True, alpha=0.1)

    # Tight bounds
    xs = []
    ys = []
    for seg, _legend in existing + future:
        for p in seg:
            try:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            except Exception:
                continue
    if xs and ys:
        # Background: approximate country outline via convex hull of all line points
        # Removed convex hull background fill to make Uzbekistan borders more visible

        margin_x = (max(xs) - min(xs)) * 0.05
        margin_y = (max(ys) - min(ys)) * 0.05
        # Add extra margin on the left (west) to show northern regions better
        margin_x_left = (max(xs) - min(xs)) * 0.15
        ax.set_xlim(min(xs) - margin_x_left, max(xs) + margin_x)
        ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    fig.savefig(OUTPUT_SVG)
    print(f"Saved {OUTPUT_PNG} and {OUTPUT_SVG}")

    # ------------------------------------------------------------------
    # Minimal interactive HTML (Leaflet) with embedded GeoJSON
    # ------------------------------------------------------------------
    try:
        with EXISTING["geojson"].open("r", encoding="utf-8") as f:
            existing_geojson = json.load(f)
        with FUTURE["geojson"].open("r", encoding="utf-8") as f:
            future_geojson = json.load(f)
        with REGION_GEOJSON.open("r", encoding="utf-8") as f:
            regions_geojson = normalize_regions_geojson(json.load(f))

        template = """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Uzbekistan Grid (Existing vs Planned)</title>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
          <style>
            html, body, #map { height: 100%; margin: 0; }
            .legend { background: white; padding: 6px 8px; font: 12px/14px Arial; }
            .legend i { width: 14px; height: 4px; float: left; margin-right: 6px; opacity: 0.9; }
          </style>
        </head>
        <body>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
        const regions = __REGIONS__;
        const existing = __EXISTING__;
        const future = __FUTURE__;
        
        const map = L.map('map');
        
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 12,
            attribution: '? OpenStreetMap'
        }).addTo(map);
        
        const regionLayer = L.geoJSON(regions, {
            style: function(feature) {
                return {color:'#bdbdbd', weight:0.6, fillColor:'#f5f5f5', fillOpacity:0.6};
            }
        }).addTo(map);
        
        function styleLine(feature) {
            const legend = ((feature.properties && feature.properties.Legend) || '').toLowerCase();
            const is500 = legend.indexOf('500') !== -1;
            const isExisting = feature._layerGroup === 'existing';
            let color, weight;
            if (isExisting) {
                color = is500 ? '#0066cc' : '#3399ff';
            } else {
                color = is500 ? '#cc0000' : '#ff6666';
            }
            weight = is500 ? 3 : 2;
            return {color: color, weight: weight, opacity: 0.8};
        }
        
        const existingLayer = L.geoJSON(existing, {
            style: function(f) { f._layerGroup='existing'; return styleLine(f); }
        }).addTo(map);
        
        const futureLayer = L.geoJSON(future, {
            style: function(f) { f._layerGroup='future'; return styleLine(f); }
        }).addTo(map);
        
        const allBounds = L.featureGroup([regionLayer, existingLayer, futureLayer]).getBounds();
        if (allBounds.isValid()) { map.fitBounds(allBounds.pad(0.05)); }
        
        const legend = L.control({position: 'topright'});
        legend.onAdd = function(map) {
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
              <div><i style="background:#0066cc;height:6px"></i>Existing 500 kV</div>
              <div><i style="background:#3399ff"></i>Existing 220 kV</div>
              <div><i style="background:#cc0000;height:6px"></i>Planned 500 kV</div>
              <div><i style="background:#ff6666"></i>Planned 220 kV</div>
            `;
            return div;
        };
        legend.addTo(map);
        </script>
        </body>
        </html>
        """
        html = template.replace('__REGIONS__', json.dumps(regions_geojson)) \
                       .replace('__EXISTING__', json.dumps(existing_geojson)) \
                       .replace('__FUTURE__', json.dumps(future_geojson))

        OUTPUT_HTML.write_text(html, encoding='utf-8')
        print(f"Saved {OUTPUT_HTML}")
    except Exception as e:
        print(f"HTML map not generated: {e}")


if __name__ == "__main__":
    main()
