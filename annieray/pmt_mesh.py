"""Shared PMT mesh loading for tracer kernel and viz server."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parent.parent / "pmt_meshes"

# Mesh type index → (filename, recenter, type_name)
PMT_BODY_SPECS: list[tuple[int, str, bool, str]] = [
    (0, "pmt_lux_bottom.gdml",    True,  "LUX"),
    (1, "pmt_etel_top.gdml",      True,  "ETEL"),
    (2, "pmt_8inch_body.gdml",    False, "Hamamatsu"),
    (3, "pmt_10inch_body.gdml",   False, "Watchboy"),
]


@dataclass
class PMTMeshData:
    """Per-type PMT mesh in local frame (centroid at origin)."""
    vertices: np.ndarray     # (V, 3) float32 — flat triangle soup
    material_ids: np.ndarray  # (T,) int32 — MaterialID per triangle
    bounding_radius: float    # mm — max ||vertex|| + 1 mm margin
    n_tris: int


def parse_gdml_flattened(path: Path, recenter: bool = True
                         ) -> tuple[np.ndarray, int]:
    """Parse a GDML tessellated mesh into (flat_vertices, n_tris)."""
    tree = ET.parse(path)
    root = tree.getroot()
    positions = root.findall(".//position")
    verts = {
        p.attrib["name"]: (float(p.attrib["x"]), float(p.attrib["y"]), float(p.attrib["z"]))
        for p in positions
    }
    triangles = root.findall(".//triangular")
    out = []
    for tri in triangles:
        for key in ("vertex1", "vertex2", "vertex3"):
            out.extend(verts[tri.attrib[key]])
    arr = np.array(out, dtype=np.float32).reshape(-1, 3)
    if recenter:
        arr -= arr.mean(axis=0)
    return arr, len(arr) // 3


def _compute_bounding_radius(vertices: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(vertices, axis=1))) + 1.0


def load_pmt_body_meshes() -> dict[int, PMTMeshData]:
    """Load all 4 PMT body meshes and classify per-triangle materials.

    Returns dict mapping mesh type index 0-3 to PMTMeshData.
    Missing files are omitted (graceful fallback).
    """
    from annieray.materials import classify_pmt_body

    result: dict[int, PMTMeshData] = {}
    for mi, gn, rc, tn in PMT_BODY_SPECS:
        p = MESH_DIR / gn
        if not p.exists():
            print(f"  PMT mesh {mi} ({gn}): NOT FOUND")
            continue
        flat, n_tris = parse_gdml_flattened(p, recenter=rc)
        tris_333 = flat.reshape(-1, 3, 3)
        mat_ids = classify_pmt_body(tris_333, tn)
        bradius = _compute_bounding_radius(flat)
        result[mi] = PMTMeshData(
            vertices=flat,
            material_ids=mat_ids,
            bounding_radius=bradius,
            n_tris=n_tris,
        )
        print(f"  PMT body mesh {mi} ({tn}): {n_tris} tris, "
              f"bounding radius {bradius:.1f} mm, "
              f"PC={int((mat_ids == 2).sum())}, "
              f"GLASS={int((mat_ids == 1).sum())}, "
              f"PVC={int((mat_ids == 3).sum())}")
    return result


def build_body_tris_arrays(
    body_meshes: dict[int, PMTMeshData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all body-mesh triangles and build offset table.

    Returns
    -------
    body_tris : (T_global, 9) float32
        Each row is 3 vertices (v0x,v0y,v0z, v1x,…, v2z) in local frame.
    body_mat_ids : (T_global,) int32
        Material ID per triangle.
    body_offsets : (5,) int32
        Start index in body_tris for each mesh type 0-3, plus sentinel.
    """
    tri_list: list[np.ndarray] = []
    mat_list: list[np.ndarray] = []
    offsets = [0]
    for mt in range(4):
        md = body_meshes.get(mt)
        if md is not None:
            tris_9 = md.vertices.reshape(-1, 9)
            tri_list.append(tris_9)
            mat_list.append(md.material_ids)
            offsets.append(offsets[-1] + md.n_tris)
        else:
            offsets.append(offsets[-1])
    body_tris = np.concatenate(tri_list, axis=0) if tri_list else np.zeros((0, 9), dtype=np.float32)
    body_mat_ids = np.concatenate(mat_list, axis=0) if mat_list else np.zeros(0, dtype=np.int32)
    body_offsets = np.array(offsets, dtype=np.int32)
    return body_tris, body_mat_ids, body_offsets
