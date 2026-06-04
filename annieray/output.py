"""Parquet output for ray tracer hit data and detector registry."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# 15-column hit schema: 13 kernel cols + arrival_time + wavelength
HIT_SCHEMA = pa.schema([
    ("hit_flag", pa.int32()),
    ("t", pa.float32()),
    ("x", pa.float32()),
    ("y", pa.float32()),
    ("z", pa.float32()),
    ("nx", pa.float32()),
    ("ny", pa.float32()),
    ("nz", pa.float32()),
    ("component_id", pa.int32()),
    ("detector_index", pa.int32()),
    ("detector_system", pa.int32()),
    ("local_u", pa.float32()),
    ("local_v", pa.float32()),
    ("arrival_time", pa.float32()),
    ("wavelength", pa.float32()),
])


def write_hits(
    hits: np.ndarray,
    path: Path,
    photon_ids: Optional[np.ndarray] = None,
) -> None:
    """Write (N, 15) hit array to Parquet.

    If hits is (N, 13) (pre-expansion), arrival_time and wavelength
    are filled with NaN.
    """
    from annieray.tracer import HI, HT, HX, HY, HZ, HNX, HNY, HNZ, HCID, HDI, HDS, HLU, HLV, N_HIT_COLS

    n = hits.shape[0]
    ncols = hits.shape[1]
    if photon_ids is None:
        photon_ids = np.arange(n, dtype=np.int64)

    # Expand to 15 cols if needed
    if ncols == N_HIT_COLS:
        full = np.full((n, 15), np.nan, dtype=np.float32)
        full[:, :N_HIT_COLS] = hits
    else:
        full = hits

    table = pa.table({
        "hit_flag": full[:, HI].astype(np.int32),
        "t": full[:, HT].astype(np.float32),
        "x": full[:, HX].astype(np.float32),
        "y": full[:, HY].astype(np.float32),
        "z": full[:, HZ].astype(np.float32),
        "nx": full[:, HNX].astype(np.float32),
        "ny": full[:, HNY].astype(np.float32),
        "nz": full[:, HNZ].astype(np.float32),
        "component_id": full[:, HCID].astype(np.int32),
        "detector_index": full[:, HDI].astype(np.int32),
        "detector_system": full[:, HDS].astype(np.int32),
        "local_u": full[:, HLU].astype(np.float32),
        "local_v": full[:, HLV].astype(np.float32),
        "arrival_time": full[:, 13].astype(np.float32),
        "wavelength": full[:, 14].astype(np.float32),
        "photon_id": photon_ids,
    })
    pq.write_table(table, str(path))


def write_detector_config(detectors: list, path: Path) -> None:
    """Write detector registry to YAML."""
    from annieray.detectors import detector_config_to_yaml
    detector_config_to_yaml(detectors, path)
