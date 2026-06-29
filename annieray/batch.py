"""Batch-mode event generation for the full ANNIE detector.

Provides:
  - Muon topology sampling (fixed, from file, or random).
  - Event loop that calls ``trace_cherenkov`` and optionally runs the
    PMT digital model.
  - ``BatchAccumulator`` for fast PyArrow-batched writes to Parquet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from annieray.tracer import (
    HI, HDI, HDS, HLU, HLV, H_ARRIVAL, H_WAVELEN,
    DET_SYS_PMT, DET_SYS_LAPPD_ANNIE, trace_cherenkov, Geometry,
)


# ---------------------------------------------------------------------------
# Schemas for output tables
# ---------------------------------------------------------------------------

PHOTON_HIT_SCHEMA = pa.schema([
    ("event_id", pa.int64()),
    ("detector_system", pa.int32()),
    ("detector_index", pa.int32()),
    ("local_u", pa.float32()),
    ("local_v", pa.float32()),
    ("arrival_time", pa.float32()),
    ("wavelength", pa.float32()),
])

PMT_RESPONSE_SCHEMA = pa.schema([
    ("event_id", pa.int64()),
    ("pmt_index", pa.int32()),
    ("charge", pa.float32()),
    ("time", pa.float32()),
    ("n_hits", pa.int32()),
])


# ---------------------------------------------------------------------------
# Muon topology sampling
# ---------------------------------------------------------------------------


def _sample_tank_position(rng: np.random.Generator, tank_radius: float,
                          tank_z_min: float, tank_z_max: float
                          ) -> tuple[float, float, float]:
    """Uniform rejection-sampled (x, y, z) inside the tank cylinder."""
    r = tank_radius * 0.9
    while True:
        x = rng.uniform(-r, r)
        y = rng.uniform(-r, r)
        if x * x + y * y <= r * r:
            break
    z = rng.uniform(tank_z_min + 100.0, tank_z_max - 100.0)
    return (float(x), float(y), float(z))


def sample_muon_state(
    event_id: int,
    config: "BatchConfig",
    rng: np.random.Generator,
    geometry: Geometry | None = None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Return ``(muon_pos, muon_dir)`` for a given event.

    *muon_pos* is a 4-tuple ``(x, y, z, t0)``.
    *muon_dir* is a 3-tuple ``(dx, dy, dz)``.
    """
    if config.muon_file is not None:
        lines = _MUON_FILE_CACHE
        idx = event_id % len(lines)
        x, y, z, t0, dx, dy, dz = lines[idx]
    elif config.muon_fixed is not None:
        x, y, z, t0, dx, dy, dz = config.muon_fixed
    else:
        # Random topology
        if geometry is None:
            x, y, z = _sample_tank_position(rng, 1524.0, 19.0, 3861.0)
        else:
            x, y, z = _sample_tank_position(
                rng, geometry.tank_radius, geometry.tank_z_min, geometry.tank_z_max
            )
        t0 = 0.0
        # Downward-going with small random scatter (≈ 5 deg)
        theta = rng.uniform(0.0, np.radians(5.0))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        sin_t = np.sin(theta)
        dx = float(sin_t * np.cos(phi))
        dy = float(sin_t * np.sin(phi))
        dz = -float(np.cos(theta))

    pos = (x, y, z, t0)
    direc = (dx, dy, dz)
    return pos, direc


# ---------------------------------------------------------------------------
# Muon-file cache
# ---------------------------------------------------------------------------

_MUON_FILE_CACHE: list[tuple[float, float, float, float, float, float, float]] = []


def _load_muon_file(path: Path) -> list:
    """Parse the topology file and populate the global cache."""
    global _MUON_FILE_CACHE
    _MUON_FILE_CACHE = []
    with open(path) as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                raise ValueError(
                    f"{path}:{line_no + 1} — expected 7 values "
                    f"(x y z t0 dx dy dz), got {len(parts)}"
                )
            _MUON_FILE_CACHE.append(tuple(float(p) for p in parts))
    if not _MUON_FILE_CACHE:
        raise ValueError(f"{path}: no valid topology lines found")
    return _MUON_FILE_CACHE


# ---------------------------------------------------------------------------
# BatchAccumulator
# ---------------------------------------------------------------------------


@dataclass
class BatchAccumulator:
    """Accumulates per-event data and writes Parquet at the end.

    Each ``append_event()`` call appends a small Arrow ``RecordBatch``.
    The final ``write()`` concatenates all batches and writes two Parquet
    files: ``photon_hits.parquet`` and (if PMT response was enabled)
    ``pmt_responses.parquet``.
    """

    photon_batches: list[pa.RecordBatch] = field(default_factory=list)
    pmt_batches: list[pa.RecordBatch] = field(default_factory=list)
    _n_photon_rows: int = 0
    _n_pmt_rows: int = 0

    # Pre-allocated arrays reused per event to avoid churn
    _photon_cols: dict = field(default_factory=dict)
    _pmt_cols: dict = field(default_factory=dict)

    def append_event(
        self,
        event_id: int,
        hits: np.ndarray,
        pmt_responses: Optional[dict[int, dict]] = None,
    ) -> None:
        """Record one event's hits and (optionally) PMT responses."""
        self._append_photon_hits(event_id, hits)
        if pmt_responses is not None:
            self._append_pmt_responses(event_id, pmt_responses)

    # ------------------------------------------------------------------
    # Photon hits
    # ------------------------------------------------------------------

    def _append_photon_hits(self, event_id: int, hits: np.ndarray) -> None:
        """Extract per-detector hit 4-vectors and append a batch."""
        # Select detector hits (PMT or ANNIE LAPPD)
        det_mask = (
            (np.abs(hits[:, HDS] - DET_SYS_PMT) < 0.5)
            | (np.abs(hits[:, HDS] - DET_SYS_LAPPD_ANNIE) < 0.5)
        )
        if not det_mask.any():
            return

        sel = hits[det_mask]
        n = sel.shape[0]

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(np.full(n, event_id, dtype=np.int64)),
                pa.array(sel[:, HDS].astype(np.int32)),
                pa.array(sel[:, HDI].astype(np.int32)),
                pa.array(sel[:, HLU]),
                pa.array(sel[:, HLV]),
                pa.array(sel[:, H_ARRIVAL]),
                pa.array(sel[:, H_WAVELEN]),
            ],
            schema=PHOTON_HIT_SCHEMA,
        )
        self.photon_batches.append(batch)
        self._n_photon_rows += n

    # ------------------------------------------------------------------
    # PMT responses
    # ------------------------------------------------------------------

    def _append_pmt_responses(
        self, event_id: int, responses: dict[int, dict]
    ) -> None:
        if not responses:
            return

        idx = sorted(responses.keys())
        n = len(idx)

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(np.full(n, event_id, dtype=np.int64)),
                pa.array(np.array(idx, dtype=np.int32)),
                pa.array(np.array([responses[i]["charge"] for i in idx], dtype=np.float32)),
                pa.array(np.array([responses[i]["time"] for i in idx], dtype=np.float32)),
                pa.array(np.array([responses[i]["n_hits"] for i in idx], dtype=np.int32)),
            ],
            schema=PMT_RESPONSE_SCHEMA,
        )
        self.pmt_batches.append(batch)
        self._n_pmt_rows += n

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, output_dir: Path) -> dict[str, Path]:
        """Concatenate all accumulated batches and write Parquet files.

        Returns ``{"photon_hits": path, "pmt_responses": path}``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        if self.photon_batches:
            table = pa.Table.from_batches(self.photon_batches, schema=PHOTON_HIT_SCHEMA)
            path = output_dir / "photon_hits.parquet"
            pq.write_table(table, str(path))
            paths["photon_hits"] = path

        if self.pmt_batches:
            table = pa.Table.from_batches(self.pmt_batches, schema=PMT_RESPONSE_SCHEMA)
            path = output_dir / "pmt_responses.parquet"
            pq.write_table(table, str(path))
            paths["pmt_responses"] = path

        return paths


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """All parameters for a batch run."""

    # Events
    n_events: int = 100
    muon_fixed: Optional[tuple[float, float, float, float, float, float, float]] = None
    muon_file: Optional[Path] = None

    # Photon generation
    photons_per_cm: int = 150
    wavelength_nm: float = 350.0
    max_bounces: int = 0

    # Response models
    pmt_response: bool = False
    pmt_full_wf: bool = False

    # I / O
    output_dir: Path = Path("results")
    record_events: bool = True

    # Reproducibility
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_batch(
    geometry: Geometry,
    config: BatchConfig,
    optics_config: Optional[dict] = None,
) -> dict[str, Path]:
    """Run N events and return paths to output files.

    Parameters
    ----------
    geometry : Geometry
        Pre-built detector geometry.
    config : BatchConfig
    optics_config : dict or None
        Per-material optical properties (for multi-bounce mode).

    Returns
    -------
    dict[str, Path]
        ``{"photon_hits": ..., "pmt_responses": ...}`` — only keys that
        were actually written.
    """
    rng = np.random.default_rng(config.seed)
    accumulator = BatchAccumulator()

    # Pre-load muon file if specified
    if config.muon_file is not None:
        _load_muon_file(config.muon_file)

    # Pre-build PMT config lookup only once
    from annieray.pmt_response import process_pmt_hits

    t_start = time.time()

    for event_id in range(config.n_events):
        muon_pos, muon_dir = sample_muon_state(
            event_id, config, rng, geometry,
        )

        hits = trace_cherenkov(
            muon_pos, muon_dir,
            photons_per_cm=config.photons_per_cm,
            geometry=geometry,
            rng=rng,
            wavelength_nm=config.wavelength_nm,
            max_bounces=config.max_bounces,
            optics_config=optics_config,
        )

        pmt_responses = None
        if config.pmt_response:
            pmt_responses = process_pmt_hits(
                hits, geometry, rng=rng, full_wf=config.pmt_full_wf,
            )

        if config.record_events:
            accumulator.append_event(event_id, hits, pmt_responses)

        if (event_id + 1) % max(1, config.n_events // 10) == 0:
            elapsed = time.time() - t_start
            rate = (event_id + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{event_id + 1}/{config.n_events}] "
                f"{elapsed:.1f}s elapsed, {rate:.1f} ev/s"
            )

    elapsed = time.time() - t_start
    print(f"  Total: {elapsed:.1f}s for {config.n_events} events "
          f"({config.n_events / elapsed:.1f} ev/s)")

    if config.record_events:
        paths = accumulator.write(config.output_dir)
        for kind, p in paths.items():
            hit_str = f" ({accumulator._n_photon_rows} photon rows)" if kind == "photon_hits" else ""
            pmt_str = f" ({accumulator._n_pmt_rows} PMT response rows)" if kind == "pmt_responses" else ""
            print(f"  Wrote {p}{hit_str}{pmt_str}")
        return paths

    return {}
