"""Pre-process scan PLY files into numpy arrays for the viz server."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

SCAN_DIR = Path(__file__).resolve().parent.parent / "scan files by part" / "transformed"
OUT_DIR = SCAN_DIR

def process_ply(name: str):
    src = SCAN_DIR / f"{name}.ply"
    if not src.exists():
        print(f"  {name}.ply: not found, skipping")
        return
    mesh = trimesh.load(str(src))
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.faces, dtype=np.int32)
    np.save(OUT_DIR / f"{name}_verts.npy", verts)
    np.save(OUT_DIR / f"{name}_tris.npy", tris)
    print(f"  {name}.ply → {name}_verts.npy ({len(verts)} verts) + {name}_tris.npy ({len(tris)} tris)")

def main():
    print("Pre-processing scan PLY files for viz server...")
    for name in ["AllPMTs", "SuperStructure", "BottomLayer", "TopLayer"]:
        process_ply(name)
    print("Done.")

if __name__ == "__main__":
    main()
