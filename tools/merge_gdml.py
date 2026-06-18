"""Merge two GDML meshes (glass + base) into a single GDML file.

Usage:
    python tools/merge_gdml.py pmt_glass_X pmt_base_X pmt_barrel_X_full
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import sys

MESH_DIR = Path("pmt_meshes")


def main():
    if len(sys.argv) != 4:
        print("Usage: merge_gdml.py <glass_name> <base_name> <output_name>")
        sys.exit(1)

    glass_name = sys.argv[1]
    base_name = sys.argv[2]
    out_name = sys.argv[3]

    glass_path = MESH_DIR / f"{glass_name}.gdml"
    base_path = MESH_DIR / f"{base_name}.gdml"
    out_path = MESH_DIR / f"{out_name}.gdml"

    if not glass_path.exists():
        print(f"Missing: {glass_path}")
        sys.exit(1)
    if not base_path.exists():
        print(f"Missing: {base_path}")
        sys.exit(1)

    def parse(path):
        tree = ET.parse(path)
        root = tree.getroot()
        positions = root.findall(".//position")
        verts = []
        for p in positions:
            verts.append(
                (float(p.attrib["x"]), float(p.attrib["y"]), float(p.attrib["z"]))
            )
        triangles = root.findall(".//triangular")
        tris = []
        for tri in triangles:
            tris.append(
                (tri.attrib["vertex1"], tri.attrib["vertex2"], tri.attrib["vertex3"])
            )
        return verts, tris

    g_verts, g_tris = parse(glass_path)
    b_verts, b_tris = parse(base_path)

    # Combine — rename vertex refs using index
    all_verts = g_verts + b_verts
    g_count = len(g_verts)
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>',
        '<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="'
        'http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">',
        "",
        "  <define>",
    ]
    for i, (x, y, z) in enumerate(all_verts):
        indent = " " * 6 if i > 0 else "    "
        lines.append(
            f'{indent}<position name="{out_name}_v{i}" unit="mm"'
            f' x="{x}" y="{y}" z="{z}"/>'
        )
    lines.append("  </define>")
    lines.append("")
    lines.append("  <solids>")
    lines.append(f'    <tessellated aunit="deg" lunit="mm" name="{out_name}">')

    all_tris = []
    for v1, v2, v3 in g_tris:
        idx1 = int(v1.split("_v")[1])
        idx2 = int(v2.split("_v")[1])
        idx3 = int(v3.split("_v")[1])
        all_tris.append((idx1, idx2, idx3))
    for v1, v2, v3 in b_tris:
        idx1 = int(v1.split("_v")[1]) + g_count
        idx2 = int(v2.split("_v")[1]) + g_count
        idx3 = int(v3.split("_v")[1]) + g_count
        all_tris.append((idx1, idx2, idx3))

    for v1, v2, v3 in all_tris:
        lines.append(
            f'      <triangular vertex1="{out_name}_v{v1}"'
            f' vertex2="{out_name}_v{v2}"'
            f' vertex3="{out_name}_v{v3}"/>'
        )
    lines.append("    </tessellated>")
    lines.append("  </solids>")
    lines.append("")
    lines.append("  <structure>")
    lines.append(f'    <volume name="{out_name}_vol">')
    lines.append('      <materialref ref="Aluminium"/>')
    lines.append(f'      <solidref ref="{out_name}"/>')
    lines.append("    </volume>")
    lines.append("  </structure>")
    lines.append("")
    lines.append('  <setup name="Default" version="1.0">')
    lines.append(f'    <world ref="{out_name}_vol"/>')
    lines.append("  </setup>")
    lines.append("")
    lines.append("</gdml>")
    out_path.write_text("\n".join(lines) + "\n")

    n_verts = len(all_verts)
    n_tris = len(all_tris)
    print(f"Merged: {glass_path.name} + {base_path.name} -> {out_path.name}")
    print(f"  Vertices: {n_verts}, Triangles: {n_tris}")


if __name__ == "__main__":
    main()
