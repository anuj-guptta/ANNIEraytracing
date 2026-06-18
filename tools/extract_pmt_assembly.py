"""Extract one representative barrel PMT assembly by spatial neighborhood.

Picks a specific glass position from the STEP file and extracts all solids
within a radius around it, producing a single merged GDML that preserves
relative spatial positions.

Usage:
    python tools/extract_pmt_assembly.py
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import cadquery as cq

STEP_PATH = Path("F10091903_-.step")
OUT_DIR = Path("pmt_meshes")

# Representative 8-inch glass position (panel 7, lower ring)
GLASS_8_POS = (-1234.1, 177.9, 1278.3)
# Representative 10-inch glass position (same panel, 11mm away)
GLASS_10_POS = (-1222.7, 178.4, 1278.4)

SEARCH_RADIUS = 350  # mm — large enough to capture base/holders


def _vec_to_tuple(v):
    return (v.x, v.y, v.z)


def _write_gdml(vertices, faces, path: Path, name: str):
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>',
        '<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="'
        'http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">',
        "",
        "  <define>",
    ]
    for i, (x, y, z) in enumerate(vertices):
        indent = " " * 6 if i > 0 else "    "
        lines.append(
            f'{indent}<position name="{name}_v{i}" unit="mm"'
            f' x="{x}" y="{y}" z="{z}"/>'
        )
    lines.append("  </define>")
    lines.append("")
    lines.append("  <solids>")
    lines.append(f'    <tessellated aunit="deg" lunit="mm" name="{name}">')
    for v1, v2, v3 in faces:
        lines.append(
            f'      <triangular vertex1="{name}_v{v1}"'
            f' vertex2="{name}_v{v2}"'
            f' vertex3="{name}_v{v3}"/>'
        )
    lines.append("    </tessellated>")
    lines.append("  </solids>")
    lines.append("")
    lines.append("  <structure>")
    lines.append(f'    <volume name="{name}_vol">')
    lines.append('      <materialref ref="Aluminium"/>')
    lines.append(f'      <solidref ref="{name}"/>')
    lines.append("    </volume>")
    lines.append("  </structure>")
    lines.append("")
    lines.append('  <setup name="Default" version="1.0">')
    lines.append(f'    <world ref="{name}_vol"/>')
    lines.append("  </setup>")
    lines.append("")
    lines.append("</gdml>")
    path.write_text("\n".join(lines) + "\n")


def classify(vol, dims):
    dx, dy, dz = dims
    sd = sorted([dx, dy, dz])
    s0, s1, s2 = sd

    if abs(vol - 41872.2) < 1:
        return "perch"
    if abs(vol - 42671) < 1:
        return "perch_plate"
    if abs(vol - 73755) < 5:
        if abs(s0 - s1) < 1:
            return "holder_square"
        if s2 > 150:
            return "holder_narrow" if dx > dy else "holder_narrow_rot"
        return "holder_other"

    if 3.7e6 < vol < 3.95e6:
        if abs(s0 - s1) < 3 and 195 < s2 < 210:
            return "glass_bulb"
        if 280 < s2 < 300:
            return "glass_slab"

    if 6.9e6 < vol < 7.0e6 and 200 < s0 < 300 and 300 < s2 < 400:
        return "base_8inch" if abs(s0 - s1) < 15 else "base_10inch"

    if abs(vol - 82584) < 100:
        return "bracket_plate"
    if abs(vol - 429337) < 100:
        return "mounting_plate"
    if abs(vol - 390598) < 100:
        return "structural_bar"
    if abs(vol - 624956) < 100:
        return "structural_bar_long"
    if abs(vol - 70506) < 100:
        return "bracket_70506"
    if abs(vol - 312478) < 100:
        return "plate_312k"

    if 100 < vol < 2000:
        return "fastener"

    return None


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading STEP file...")
    assembly = cq.importers.importStep(str(STEP_PATH))
    compound = assembly.val()
    solids = list(compound.Solids())
    print(f"  {len(solids)} solids loaded")

    for label, anchor, out_name in [
        ('8" barrel', GLASS_8_POS, "pmt_assembly_8inch"),
        ('10" barrel', GLASS_10_POS, "pmt_assembly_10inch"),
    ]:
        cx, cy, cz = anchor
        print(f"\n=== {label} anchor=({cx:.1f}, {cy:.1f}, {cz:.1f}) ===")

        # Find nearby solids — exclude tiny fasteners (vol < 100 mm³)
        nearby = []
        for solid in solids:
            b = solid.BoundingBox()
            scx = (b.xmin + b.xmax) / 2.0
            scy = (b.ymin + b.ymax) / 2.0
            scz = (b.zmin + b.zmax) / 2.0
            d = math.hypot(scx - cx, scy - cy, scz - cz)
            if d < SEARCH_RADIUS:
                vol = solid.Volume()
                if vol < 100:
                    continue  # skip tiny fasteners
                dims = [b.xmax - b.xmin, b.ymax - b.ymin, b.zmax - b.zmin]
                tag = classify(vol, dims) or "unknown"
                if tag in ("fastener", "unknown"):
                    continue  # skip small hardware
                nearby.append({
                    "solid": solid,
                    "vol": round(vol, 1),
                    "dims": [round(d, 1) for d in dims],
                    "centroid": [round(scx, 1), round(scy, 1), round(scz, 1)],
                    "dist": round(d, 1),
                    "tag": tag,
                })

        # Group and report
        groups = defaultdict(list)
        for s in nearby:
            groups[s["tag"]].append(s)

        print(f"  {len(nearby)} solids within {SEARCH_RADIUS}mm:")
        for tag in sorted(groups.keys()):
            items = groups[tag]
            if len(items) <= 3:
                for i in items:
                    print(f"    {tag:20s} d={i['dist']:5.1f} vol={i['vol']:>8.0f} dims={i['dims']} pos={i['centroid']}")
            else:
                vols = set(i["vol"] for i in items)
                print(f"    {tag:20s} {len(items)} instances, volumes={sorted(vols)}")

        # Tessellate and merge
        all_verts, all_faces, offset = [], [], 0
        for s in nearby:
            v_raw, t_raw = s["solid"].tessellate(0.5)
            verts = np.array([_vec_to_tuple(v) for v in v_raw], dtype=np.float64)
            tris = np.array(t_raw, dtype=np.int64) + offset
            all_verts.append(verts)
            all_faces.append(tris)
            offset += len(verts)

        merged_verts = np.vstack(all_verts) if all_verts else np.empty((0, 3))
        merged_faces = np.vstack(all_faces) if all_faces else np.empty((0, 3))

        out_path = OUT_DIR / f"{out_name}.gdml"
        print(f"  Writing {len(merged_verts)} verts, {len(merged_faces)} tris -> {out_path.name}")
        _write_gdml(merged_verts, merged_faces, out_path, out_name)

        meta_path = OUT_DIR / f"{out_name}_meta.json"
        meta = {}
        for tag, items in groups.items():
            meta[tag] = [{"vol": i["vol"], "dims": i["dims"], "centroid": i["centroid"], "dist": i["dist"]} for i in items]
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  Metadata -> {meta_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
