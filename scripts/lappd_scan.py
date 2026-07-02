"""Plot LAPPD hit counts as a 2D heatmap over muon vertex (x, z).

Assumes the batch was run with a regular grid of muon positions
(via --muon-file) so that unique x and z values form a rectangular
grid.  Produces one panel per ANNIE LAPPD (3 panels).

Usage:
    python scripts/lappd_scan.py results/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def load_data(output_dir: Path):
    hits = pq.read_table(str(output_dir / "photon_hits.parquet")).to_pandas()
    muons = pq.read_table(str(output_dir / "muon_truth.parquet")).to_pandas()
    det = pd.read_csv(output_dir / "detectors.csv")
    return hits, muons, det


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <batch_output_dir>")
        return

    output_dir = Path(sys.argv[1])
    hits, muons, det = load_data(output_dir)

    # Identify ANNIE LAPPDs (system_code == 2 in detectors.csv)
    lappd_det = det[det["system_code"] == 2].copy()
    if lappd_det.empty:
        print("No ANNIE LAPPDs found in detector registry.")
        return

    lappd_indices = lappd_det["detector_index"].values
    lappd_labels = lappd_det["label"].values
    n_lappds = len(lappd_indices)
    print(f"Found {n_lappds} ANNIE LAPPDs: indices {lappd_indices}")

    # Per-event LAPPD hit counts: (event_id, detector_index) -> n_hits
    lappd_hits = hits[hits["detector_system"] == 2]
    if lappd_hits.empty:
        print("No LAPPD hits in data.")
        return

    hit_counts = lappd_hits.groupby(["event_id", "detector_index"]).size()

    # Build a grid from muon positions
    xs = muons["pos_x"].values
    zs = muons["pos_z"].values
    u_x = np.unique(xs)
    u_z = np.unique(zs)

    if len(u_x) * len(u_z) != len(muons):
        print("Warning: muon positions do not form a complete rectangular grid "
              f"({len(u_x)} x {len(u_z)} = {len(u_x)*len(u_z)}, "
              f"expected {len(muons)}).  Falling back to scatter plot.")
        gridded = False
    else:
        gridded = True
        nx, nz = len(u_x), len(u_z)
        # Sort so heatmap axes are ascending
        u_x.sort()
        u_z.sort()
        print(f"Grid: {nx} x {nz} = {nx * nz} events")

    # Build 2D arrays of hit counts per LAPPD
    if gridded:
        x_idx = {v: i for i, v in enumerate(u_x)}
        z_idx = {v: i for i, v in enumerate(u_z)}
        maps = []
        for li in lappd_indices:
            z2d = np.full((nz, nx), np.nan)
            for _, row in muons.iterrows():
                ev = int(row["event_id"])
                xi, zi = x_idx[row["pos_x"]], z_idx[row["pos_z"]]
                z2d[zi, xi] = hit_counts.get((ev, li), 0)
            maps.append(z2d)

    # Plot
    fig, axes = plt.subplots(1, n_lappds, figsize=(6 * n_lappds, 5),
                             squeeze=False)
    axes = axes[0]

    vmin, vmax = 0, 0
    if gridded:
        vmax = max(m.max() for m in maps) if maps else 1

    for i, li in enumerate(lappd_indices):
        ax = axes[i]
        label = lappd_labels[i] if i < len(lappd_labels) else f"LAPPD {li}"

        if gridded:
            im = ax.pcolormesh(u_x, u_z, maps[i], shading="auto",
                               cmap="plasma", vmin=vmin, vmax=vmax)
            ax.set_xlabel("muon start x (mm)")
            ax.set_ylabel("muon start z (mm)")
        else:
            # Fallback: scatter
            vals = []
            for _, row in muons.iterrows():
                ev = int(row["event_id"])
                vals.append(hit_counts.get((ev, li), 0))
            sc = ax.scatter(xs, zs, c=vals, cmap="plasma", s=40,
                            edgecolors="white", linewidth=0.3)
            ax.set_xlabel("muon start x (mm)")
            ax.set_ylabel("muon start z (mm)")

        ax.set_title(f"{label}\nindex={li}")
        ax.set_aspect("equal")

        if gridded:
            fig.colorbar(im, ax=ax, label="LAPPD hits")
        else:
            fig.colorbar(sc, ax=ax, label="LAPPD hits")

    fig.suptitle("LAPPD hit counts vs muon vertex (x, z)", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
