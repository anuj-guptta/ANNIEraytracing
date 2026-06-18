"""Extract separated 8" and 10" barrel PMT assemblies.

10" assembly: Assem1.step mounting hardware + 10" glass from main STEP
8"  assembly: main STEP neighbors near 8" glass position

Usage:
    python tools/extract_barrel_assemblies.py
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import cadquery as cq

STEP_PATH = Path("F10091903_-.step")
ASSEM1_PATH = Path("Assem1.step")
OUT_DIR = Path("pmt_meshes")


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


def extract_10inch_assembly():
    """Extract complete 10" PMT assembly from Assem1.step (all 89 solids)."""
    print("\n=== 10-inch Barrel PMT Assembly ===")

    print("Loading Assem1.step...")
    assembly = cq.importers.importStep(str(ASSEM1_PATH))
    compound = assembly.val()
    solids = list(compound.Solids())
    print(f"  {len(solids)} solids")

    # Classify and collect ALL solids
    parts = []
    for solid in solids:
        vol = solid.Volume()
        b = solid.BoundingBox()
        dims = [b.xmax - b.xmin, b.ymax - b.ymin, b.zmax - b.zmin]
        cx = (b.xmin + b.xmax) / 2
        cy = (b.ymin + b.ymax) / 2
        cz = (b.zmin + b.zmax) / 2
        if abs(vol - 6941183) < 100:
            tag = "pmt_r7081"
        elif abs(vol - 73756) < 10:
            tag = "holder"
        elif abs(vol - 70506) < 10:
            tag = "bracket70"
        elif abs(vol - 112050) < 100:
            tag = "thin_plate_6mm"
        elif abs(vol - 112314) < 100:
            tag = "thin_plate_8mm"
        elif abs(vol - 42671) < 10:
            tag = "perch_plate"
        elif abs(vol - 41872) < 10:
            tag = "perch"
        elif abs(vol - 7048) < 10:
            tag = "unk_7048"
        elif abs(vol - 8120) < 10:
            tag = "unk_8120"
        elif vol < 1000:
            tag = "fastener"
        else:
            tag = f"other_{int(vol)}"
        parts.append({
            "solid": solid,
            "tag": tag,
            "vol": vol,
            "dims": dims,
            "center": (cx, cy, cz),
        })

    # Report
    groups = defaultdict(list)
    for p in parts:
        groups[p["tag"]].append(p)
    print(f"  All {len(parts)} solids:")
    for tag in sorted(groups.keys()):
        items = groups[tag]
        sample = items[0]
        print(f"    {tag:20s} {len(items):2d} instances  vol={sample['vol']:.0f}  dims=({sample['dims'][0]:.1f}, {sample['dims'][1]:.1f}, {sample['dims'][2]:.1f})")

    # Tessellate ALL solids
    print("\n  Tessellating...")
    all_verts, all_faces, offset = [], [], 0
    for p in parts:
        v_raw, t_raw = p["solid"].tessellate(0.5)
        verts = np.array([(v.x, v.y, v.z) for v in v_raw], dtype=np.float64)
        tris = np.array(t_raw, dtype=np.int64) + offset
        all_verts.append(verts)
        all_faces.append(tris)
        offset += len(verts)

    merged_verts = np.vstack(all_verts) if all_verts else np.empty((0, 3))
    merged_faces = np.vstack(all_faces) if all_faces else np.empty((0, 3))
    print(f"  Merged: {len(merged_verts)} verts, {len(merged_faces)} tris")

    # Center at origin for clean visualization
    centroid = merged_verts.mean(axis=0)
    merged_verts -= centroid
    print(f"  Centering offset: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

    out_path = OUT_DIR / "pmt_assembly_10inch.gdml"
    print(f"  Writing -> {out_path.name}")
    _write_gdml(merged_verts, merged_faces, out_path, "pmt_assembly_10inch")

    # Write body/hardware grouped GDML for togglable viz
    print("\n  Writing grouped GDML files for viz...")
    body_groups = {"pmt_r7081": groups["pmt_r7081"]}
    hw_groups = {k: v for k, v in groups.items() if k != "pmt_r7081"}
    for out_name, subset in [("pmt_10inch_body", body_groups), ("pmt_10inch_hardware", hw_groups)]:
        t_verts, t_faces, t_offset = [], [], 0
        for tag in sorted(subset.keys()):
            for p in subset[tag]:
                v_raw, t_raw = p["solid"].tessellate(0.5)
                verts = np.array([(v.x, v.y, v.z) for v in v_raw], dtype=np.float64)
                tris = np.array(t_raw, dtype=np.int64) + t_offset
                verts -= centroid
                t_verts.append(verts)
                t_faces.append(tris)
                t_offset += len(verts)
        tv = np.vstack(t_verts) if t_verts else np.empty((0, 3))
        tf = np.vstack(t_faces) if t_faces else np.empty((0, 3))
        type_path = OUT_DIR / f"{out_name}.gdml"
        _write_gdml(tv, tf, type_path, out_name)
        print(f"    {out_name:25s} -> {type_path.name} ({len(tv)} verts, {len(tf)} tris)")

    return merged_verts, merged_faces


def classify_main(vol, dims, center_z):
    sd = sorted(dims)
    if 3.88e6 < vol < 3.95e6:
        return "glass_10inch"
    if 3.73e6 < vol < 3.75e6:
        return "glass_8inch"
    if abs(vol - 429337) < 100:
        return "mounting_plate"
    if abs(vol - 82584) < 100:
        return "bracket_82"
    if abs(vol - 73755) < 10:
        return "holder"
    if abs(vol - 70506) < 100:
        return "bracket70"
    if abs(vol - 41872) < 10:
        return "perch"
    if abs(vol - 42671) < 10:
        return "perch_plate"
    if abs(vol - 112304) < 100 or abs(vol - 112050) < 100:
        return "thin_plate"
    if 6.9e6 < vol < 7.0e6:
        return "body_10inch"
    if vol > 1000:
        return None  # skip large structural elements
    return "fastener"


def extract_8inch_assembly():
    """Extract 8-inch PMT assembly from main STEP using spatial neighborhood."""
    print("\n=== 8-inch Barrel PMT Assembly ===")

    # Representative 8-inch glass position
    GLASS_8_POS = (-1234.1, 177.9, 1278.3)
    SEARCH_RADIUS = 350

    print("Loading main STEP...")
    assembly = cq.importers.importStep(str(STEP_PATH))
    compound = assembly.val()
    solids = list(compound.Solids())
    print(f"  {len(solids)} solids")

    cx, cy, cz = GLASS_8_POS
    print(f"  Anchor: ({cx:.1f}, {cy:.1f}, {cz:.1f})")

    nearby = []
    for solid in solids:
        b = solid.BoundingBox()
        scx = (b.xmin + b.xmax) / 2
        scy = (b.ymin + b.ymax) / 2
        scz = (b.zmin + b.zmax) / 2
        d = math.hypot(scx - cx, scy - cy, scz - cz)
        if d < SEARCH_RADIUS:
            vol = solid.Volume()
            dims = [b.xmax - b.xmin, b.ymax - b.ymin, b.zmax - b.zmin]
            tag = classify_main(vol, dims, scz)
            if tag is None or tag == "fastener":
                continue
            nearby.append({
                "solid": solid,
                "tag": tag,
                "vol": round(vol, 1),
                "dims": [round(d, 1) for d in dims],
                "center": [round(scx, 1), round(scy, 1), round(scz, 1)],
                "dist": round(d, 1),
            })

    # Keep only core 8-inch components (holder at d=295 is a 10" remnant)
    CORE_TAGS = {"glass_8inch", "mounting_plate", "bracket_82"}
    nearby = [s for s in nearby if s["tag"] in CORE_TAGS]

    # Report
    groups = defaultdict(list)
    for s in nearby:
        groups[s["tag"]].append(s)

    print(f"  {len(nearby)} core 8-inch solids:")
    for tag in sorted(groups.keys()):
        items = groups[tag]
        for i in items:
            print(f"    {tag:20s} d={i['dist']:5.1f} vol={i['vol']:>8.0f}")

    # Tessellate
    all_verts, all_faces, offset = [], [], 0
    for s in nearby:
        v_raw, t_raw = s["solid"].tessellate(0.5)
        verts = np.array([(v.x, v.y, v.z) for v in v_raw], dtype=np.float64)
        tris = np.array(t_raw, dtype=np.int64) + offset
        all_verts.append(verts)
        all_faces.append(tris)
        offset += len(verts)

    merged_verts = np.vstack(all_verts) if all_verts else np.empty((0, 3))
    merged_faces = np.vstack(all_faces) if all_faces else np.empty((0, 3))

    # Center at origin
    centroid = merged_verts.mean(axis=0)
    merged_verts -= centroid
    print(f"\n  Centering offset: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

    out_path = OUT_DIR / "pmt_assembly_8inch.gdml"
    print(f"  Writing {len(merged_verts)} verts, {len(merged_faces)} tris -> {out_path.name}")
    _write_gdml(merged_verts, merged_faces, out_path, "pmt_assembly_8inch")

    # Write body/hardware grouped GDML for togglable viz
    print("\n  Writing grouped GDML files for viz...")
    body_groups = {"glass_8inch": groups["glass_8inch"]}
    hw_groups = {k: v for k, v in groups.items() if k != "glass_8inch"}
    for out_name, subset in [("pmt_8inch_body", body_groups), ("pmt_8inch_hardware", hw_groups)]:
        t_verts, t_faces, t_offset = [], [], 0
        for tag in sorted(subset.keys()):
            for s in subset[tag]:
                v_raw, t_raw = s["solid"].tessellate(0.5)
                verts = np.array([(v.x, v.y, v.z) for v in v_raw], dtype=np.float64)
                tris = np.array(t_raw, dtype=np.int64) + t_offset
                verts -= centroid
                t_verts.append(verts)
                t_faces.append(tris)
                t_offset += len(verts)
        tv = np.vstack(t_verts) if t_verts else np.empty((0, 3))
        tf = np.vstack(t_faces) if t_faces else np.empty((0, 3))
        type_path = OUT_DIR / f"{out_name}.gdml"
        _write_gdml(tv, tf, type_path, out_name)
        print(f"    {out_name:25s} -> {type_path.name} ({len(tv)} verts, {len(tf)} tris)")

    # Write metadata
    meta = {}
    for tag, items in groups.items():
        meta[tag] = [
            {"vol": i["vol"], "dims": i["dims"], "center": i["center"], "dist": i["dist"]}
            for i in items
        ]
    meta_path = OUT_DIR / "pmt_assembly_8inch_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata -> {meta_path.name}")

    return merged_verts, merged_faces


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 10-inch assembly
    v10, f10 = extract_10inch_assembly()

    # 8-inch assembly
    v8, f8 = extract_8inch_assembly()

    print("\nDone.")
    if v10 is not None and v8 is not None:
        print(f"  10-inch: {len(v10)} verts, {len(f10)} tris")
        print(f"  8-inch:  {len(v8)} verts, {len(f8)} tris")


if __name__ == "__main__":
    main()
