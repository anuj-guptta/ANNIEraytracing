"""Reconstruct a closed contiguous mesh from the GDML triangle soup.

Reads PHASE2_INNER_STRUCTURE.gdml (disconnected laser-scan triangle soup),
merges duplicate vertices, and writes PHASE2_INNER_STRUCTURE_closed.gdml.

The original has 298k vertices (all unique per-triangle), which trimesh
collapses to ~49k unique positions. After dedup, every edge is shared
by 2+ faces — zero boundary edges — but ~0.7% of edges are non-manifold
(shared by 3-4 faces at structural junctions). This is fine for ray tracing:
the surface is geometrically closed and continuous.

Usage:
    python tools/reconstruct_mesh.py
"""

import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from pathlib import Path

GDML_PATH = Path("PHASE2_INNER_STRUCTURE.gdml")
OUT_PATH = Path("PHASE2_INNER_STRUCTURE_closed.gdml")

_MATERIAL = """    <material name="StainlessSteel0x3966010" state="solid">
      <T name="Rhodamine6G" fractionmax="0.01" fractionmin="0"/>
      <P unit="mg/cm3" value="1.62e-06"/>
      <D unit="g/cm3" value="7.93"/>
      <composite n="1" ref="Fe"/>
    </material>"""


def parse_gdml(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    positions = root.findall(".//position")
    vertices = np.array(
        [(float(p.attrib["x"]), float(p.attrib["y"]), float(p.attrib["z"]))
         for p in positions],
        dtype=np.float64,
    )
    triangles = root.findall(".//triangular")
    vname_to_idx = {p.attrib["name"]: i for i, p in enumerate(positions)}
    faces = np.array(
        [(vname_to_idx[t.attrib["vertex1"]],
          vname_to_idx[t.attrib["vertex2"]],
          vname_to_idx[t.attrib["vertex3"]])
         for t in triangles],
        dtype=np.int64,
    )
    return vertices, faces


def write_gdml(vertices, faces, path: Path):
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
            f'{indent}<position name="stl_v{i}" unit="mm"'
            f' x="{x}" y="{y}" z="{z}"/>'
        )
    lines.append("  </define>")
    lines.append(_MATERIAL)
    lines.append("")
    lines.append("  <solids>")
    lines.append(
        '    <tessellated aunit="deg" lunit="mm"'
        ' name="INNER_STRUCTURE_CLOSED">'
    )
    for v1, v2, v3 in faces:
        lines.append(
            f'      <triangular vertex1="stl_v{v1}" vertex2="stl_v{v2}"'
            f' vertex3="stl_v{v3}"/>'
        )
    lines.append("    </tessellated>")
    lines.append("  </solids>")
    lines.append("")
    lines.append("  <structure>")
    lines.append('    <volume name="inner_structure_closed">')
    lines.append('      <materialref ref="StainlessSteel0x3966010"/>')
    lines.append('      <solidref ref="INNER_STRUCTURE_CLOSED"/>')
    lines.append("    </volume>")
    lines.append("  </structure>")
    lines.append("")
    lines.append('  <setup name="Default" version="1.0">')
    lines.append('    <world ref="inner_structure_closed"/>')
    lines.append("  </setup>")
    lines.append("")
    lines.append("</gdml>")
    path.write_text("\n".join(lines) + "\n")


def main():
    print("Parsing GDML...")
    verts, faces = parse_gdml(GDML_PATH)
    print(f"  {len(verts)} vertices, {len(faces)} faces")

    print("Creating trimesh with process=True (dedup, merge)...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    print(f"  After process: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print("Fixing normals (consistent orientation)...")
    mesh.fix_normals()

    edges = mesh.edges
    inv = mesh.edges_unique_inverse
    counts = np.bincount(inv, minlength=len(mesh.edges_unique))
    boundary = (counts == 1).sum()
    nonmanifold = (counts > 2).sum()
    shared2 = (counts == 2).sum()
    print(f"  Unique edges: {len(mesh.edges_unique)}")
    print(f"  Shared by 2 faces: {shared2}")
    print(f"  Shared by 3+ faces (non-manifold): {nonmanifold}")
    print(f"  Boundary edges: {boundary}")
    print(f"  is_watertight: {mesh.is_watertight}")

    if boundary == 0 and nonmanifold > 0:
        print("  -> Mesh is closed (zero boundary) but has non-manifold edges.")
    elif boundary > 0:
        print("  -> WARNING: mesh has boundary edges — not fully closed!")

    print(f"\nWriting {OUT_PATH}...")
    write_gdml(mesh.vertices, mesh.faces, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
