# ANNIEraytracing

GPU-accelerated ray tracer for the ANNIE Phase II water Cherenkov detector at Fermilab. Replaces Geant4's stepwise photon integration with analytic ray-geometry intersections on the GPU using Taichi.

## Physics Context

ANNIE (Accelerator Neutrino Neutron Interaction Experiment) measures final-state neutron multiplicity from CC neutrino interactions on oxygen. Phase II is a 26-ton Gd-doped water Cherenkov detector with 132 PMTs, 5 LAPPDs, Front Muon Veto, and Muon Range Detector. This tool simulates Cherenkov photon propagation through the detector geometry to predict PMT/LAPPD hit patterns.

Key physics:
- Cherenkov photons generated from muon tracks (cos θ = 1/nβ)
- Multi-bounce optics: Fresnel reflection/transmission, diffuse reflection
- Water attenuation: absorption + Rayleigh scattering
- PMT digital model: SPE charge + transit time spread
- LAPPD readout: strip-aligned local coordinates

## Project Structure

```
annieray/
├── annieray/          # Main package (28 Python files)
│   ├── cli.py         # CLI entry point (batch, fit, viz-server, etc.)
│   ├── tracer.py      # Core: Geometry dataclass, Taichi GPU trace_kernel
│   ├── batch.py       # Batch-mode event loop, HDF5 output
│   ├── cherenkov.py   # Cherenkov cone / isotropic photon generator
│   ├── fitting.py     # Direction grid scan, observed event loader
│   ├── likelihood.py  # Poisson charge + Gaussian time residual LL
│   ├── gdml_parser.py # GDML XML → vertex/triangle arrays
│   ├── pmt_loader.py  # PMT CSV parser, coordinate transforms
│   ├── pmt_mesh.py    # PMT body/hardware mesh loading
│   ├── step_parser.py # STEP CAD → component manifest
│   ├── detectors.py   # DetectorInfo dataclass, YAML I/O
│   ├── optics.py      # Optical material database, Fresnel evaluation
│   ├── pmt_response.py  # PMT digital model (fast + full waveform)
│   ├── lappd_response.py # LAPPD digitization (Taichi-accelerated)
│   ├── lappd_model.py  # LAPPD housing geometry builder
│   ├── viz_server.py  # Interactive Three.js 3D viewer
│   └── io_h5.py       # HDF5 table I/O
├── scripts/           # Analysis/plotting utilities (12 files)
├── tools/             # Developer/debugging utilities (23 files)
├── docs/              # Documentation, thesis PDFs
├── pmt_meshes/        # PMT body/hardware meshes (GDML + numpy cache)
├── extern/LApyPD/     # Git submodule (LAPPD physics model)
├── results/           # Batch simulation output (HDF5, parquet)
└── PHASE2_INNER_STRUCTURE_closed.gdml  # Primary detector geometry
```

## Build & Run

```bash
# Install (editable mode)
pip install -e .

# Run tests
pytest

# Batch simulation
python -m annieray batch \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 \
    --events 100 --photons-per-cm 150

# Batch simulation without inner structure (PMTs + LAPPDs only)
python -m annieray batch --no-gdml \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 \
    --events 100

# Batch simulation without PMT holder meshes (faster, PMT positions still loaded)
python -m annieray batch --no-pmt-holders \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 \
    --events 100

# Direction fit
python -m annieray fit output.h5 --event 0 --show

# 3D viewer
python -m annieray viz-server --pmt-csv PMTPositions_Scan.txt --surfboard 3 --port 8080
```

### CLI Flags for Geometry Control

| Flag | Default | Description |
|------|---------|-------------|
| `--no-gdml` | false | Skip inner structure GDML mesh (tank + PMTs only) |
| `--no-pmt-holders` | false | Skip PMT body and hardware holder meshes (PMT positions still loaded) |
| `--no-lappd` | false | Skip LAPPD rectangles |
| `--surfboard` | 0 | Number of obscurant PVC surfboards (0, 1, or 3) — use 3 for Phase II |

## Key Data Files

| File | Description |
|------|-------------|
| `PHASE2_INNER_STRUCTURE_closed.gdml` | Primary GDML detector geometry mesh |
| `PMTPositions_Scan.txt` | 132 PMT positions (TSV, 9 columns) |
| `component_manifest.json` | Cached STEP parser output (PMT/LAPPD positions) |
| `corrections.csv` | LAPPD correction data |
| `MuonStartsAndDirecs.txt` | Muon start positions/directions |

## Conventions

- **Python 3.11+** with type hints
- **Taichi** for GPU kernels (`@ti.kernel`, `@ti.func`)
- **NumPy** for CPU-side array operations
- **HDF5** (h5py) for batch output
- **GDML** (lxml) for detector geometry parsing
- **CadQuery** for STEP CAD file parsing
- CLI via `python -m annieray <command>`
- No heavy frameworks — keep dependencies minimal

## Muon Direction Conventions

- **θ (theta)**: polar angle from vertical. 0° = downward (−z), 90° = horizontal, 180° = upward (+z)
- **φ (phi)**: azimuthal in XY plane. 0° = +x, 90° = +y

## Current State

The core ray tracer is complete and working. All major features implemented:
- GDML and STEP geometry parsing
- Taichi GPU ray tracing with BVH acceleration
- Cherenkov photon generation (vectorized)
- Batch-mode simulation with HDF5 output
- PMT and LAPPD digital response models
- Multi-bounce Fresnel optics
- Direction fitting framework with grid scanning
- Interactive 3D visualization server

Recent work focuses on validation, analysis scripts, and fitting accuracy.

## Testing

```bash
pytest                    # Run all tests
pytest tests/test_tracer.py  # Tracer-specific tests
```

## Git Submodules

```bash
git submodule update --init --recursive
```

## Tips for New Sessions

- The project is already mature — ask what the user wants to work on rather than starting from scratch
- GDML parsing is in `gdml_parser.py`, PMT loading in `pmt_loader.py`
- The Taichi GPU kernel is in `tracer.py` (`trace_kernel`)
- Batch output goes to `results/` as HDF5 files
- Use `@` to reference files when discussing specific code
- PDFs in `docs/` can be read with `pymupdf` (already installed)
