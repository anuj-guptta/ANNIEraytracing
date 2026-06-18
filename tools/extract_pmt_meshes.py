"""Extract PMT component meshes from the STEP CAD file.

Usage:
    python tools/extract_pmt_meshes.py
"""

import json
from pathlib import Path

import numpy as np
import cadquery as cq

STEP_PATH = Path("F10091903_-.step")
OUT_DIR = Path("pmt_meshes")


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
    lines.append(
        f'    <tessellated aunit="deg" lunit="mm" name="{name}">'
    )
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


def _classify(vol: float, dims, sorted_dims, cz: float):
    """Classify a solid by volume, dimensions, and Z position."""
    if abs(vol - 41872.2) < 1:
        return "perch"

    # PMT glass bulbs
    if 3.7e6 < vol < 3.75e6:
        return "glass_8inch"
    if 3.88e6 < vol < 3.95e6:
        return "glass_10inch"

    # Barrel PMT bases (permanently attached to glass)
    if 6.9e6 < vol < 7.0e6:
        s0, s1, s2 = sorted_dims
        if 200 < s0 < 300 and 300 < s2 < 400:
            if abs(s0 - s1) < 15:  # Type A: ~254×254×340
                return "base_8inch"
            else:                  # Type B: ~254×311×338
                return "base_10inch"

    # Barrel PMT holders (brackets that mount base+glass to structure)
    if 73700 < vol < 73800:
        dx, dy, dz = dims
        sd = sorted_dims
        if abs(sd[0] - sd[1]) < 1:  # square: ~66×66×165
            return "holder_square"
        elif sd[2] > 150 and sd[2] < 170:  # narrow: ~25×68×165
            if dy > dx:
                return "holder_narrow"
            else:
                return "holder_narrow_rotated"
        return None

    # Bottom housings (LUX — complete glued unit)
    if 1.05e7 < vol < 1.07e7 and cz < 500:
        return "lux_bottom"       # 2 instances, ~253×253×374
    if 1.07e7 < vol < 1.10e7 and cz < 500:
        return "lux_bottom_wide"  # 12 instances, ~316×481×367 — with wings

    # Top housing (ETEL — complete glued unit)
    if 1.16e7 < vol < 1.19e7 and cz > 3500:
        return "etel_top"

    return None


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading STEP file...")
    assembly = cq.importers.importStep(str(STEP_PATH))
    compound = assembly.val()
    solids = list(compound.Solids())
    print(f"  {len(solids)} solids loaded")

    # Classify and pick one instance of each type
    types = {}
    for solid in solids:
        b = solid.BoundingBox()
        vol = solid.Volume()
        dx, dy, dz = b.xmax - b.xmin, b.ymax - b.ymin, b.zmax - b.zmin
        dims = [dx, dy, dz]
        cz = (b.zmin + b.zmax) / 2.0
        sorted_d = sorted([dx, dy, dz])

        label = _classify(vol, dims, sorted_d, cz)
        if label and label not in types:
            types[label] = {
                "solid": solid,
                "count": 0,
                "vol": round(vol, 1),
                "dims": [round(dx, 1), round(dy, 1), round(dz, 1)],
                "center": [round((b.xmin + b.xmax) / 2, 1),
                           round((b.ymin + b.ymax) / 2, 1),
                           round(cz, 1)],
            }
        if label:
            types[label]["count"] += 1

    print(f"\nFound {len(types)} unique types:")
    for label, info in sorted(types.items()):
        print(f"  {label}: {info['count']} instances,"
              f" vol={info['vol']:.0f}, dims={info['dims']}")
        print(f"    center={info['center']}")

    # Tessellate and save each type
    for label, info in sorted(types.items()):
        print(f"\nTessellating {label}...")
        verts_raw, tris = info["solid"].tessellate(0.5)
        vertices = np.array([_vec_to_tuple(v) for v in verts_raw],
                            dtype=np.float64)
        faces = np.array(tris, dtype=np.int64)
        print(f"  {len(vertices)} vertices, {len(faces)} faces")

        gdml_name = f"pmt_{label}"
        gdml_path = OUT_DIR / f"{gdml_name}.gdml"
        _write_gdml(vertices, faces, gdml_path, gdml_name)
        print(f"  -> {gdml_path}")

    # Write metadata
    meta = {}
    for label, info in sorted(types.items()):
        meta[label] = {
            "count": info["count"],
            "volume": info["vol"],
            "bounding_box_dims": info["dims"],
            "sample_center": info["center"],
        }

    meta_path = OUT_DIR / "pmt_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nMetadata -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
