# ANNIEraytracing

GPU-accelerated ray tracer for the ANNIE detector, implemented in Taichi.
Traces photons through a GDML-detailed inner structure, PMTs, and LAPPDs,
recording hit positions, local coordinates, arrival times, and wavelengths.

## Quick Start

```bash
# Uniform mode: random photon directions inside the tank
python -m annieray run --gdml PHASE2_INNER_STRUCTURE.gdml \
    --pmt-csv PMTPositions_Scan.txt \
    --photons 100000 --mode uniform -o hits.parquet

# Cherenkov mode: photons from a muon vertex on a Cherenkov cone
python -m annieray run --gdml PHASE2_INNER_STRUCTURE.gdml \
    --pmt-csv PMTPositions_Scan.txt \
    --photons 100000 --mode cherenkov --wavelength 350 -o hits.parquet

# Build detector registry YAML (one-time setup for model coupling)
python -m annieray build-detector-config --gdml PHASE2_INNER_STRUCTURE.gdml \
    --pmt-csv PMTPositions_Scan.txt -o detectors.yaml

# Interactive 3D viewer
python -m annieray viz-server --gdml PHASE2_INNER_STRUCTURE.gdml \
    --pmt-csv PMTPositions_Scan.txt --port 8080
```

Both commands accept `--lappd-model annie` to replace one default LAPPD
with the physically correct Kandemir housing model, and `--lappd-indices`
to choose which LAPPD candidates from the STEP manifest are active.

## Batch-Mode Simulations

The `run` command requires no display server or browser. It loads the
geometry, generates photons on the CPU, transfers them to the GPU kernel,
and writes results to Parquet.

### Output schema (15 columns)

| Column           | Type    | Description                               |
|------------------|---------|-------------------------------------------|
| hit_flag         | int32   | 1 = hit something, 0 = missed everything  |
| t                | float32 | Path length from origin to hit (mm)       |
| x, y, z          | float32 | Hit position in detector coordinates (mm) |
| nx, ny, nz       | float32 | Normal at hit point (outward-facing)      |
| component_id     | int32   | 1=structure, 2=PMT, 3=LAPPD, 4=tank wall |
| detector_index   | int32   | Index into detector registry, -1 if none  |
| detector_system  | int32   | 0=pmt, 1=lappd_default, 2=lappd_annie    |
| local_u          | float32 | PMT: polar angle from direction (rad); LAPPD: position along strips (mm) |
| local_v          | float32 | PMT: azimuthal angle around direction (rad); LAPPD: position across strips (mm) |
| arrival_time     | float32 | Photon travel time (ns), NaN in uniform mode |
| wavelength       | float32 | Photon wavelength (nm), NaN in uniform mode |
| photon_id        | int64   | Sequential photon index (0..N-1)          |

### Advanced options

```bash
# Control random seed for reproducibility
python -m annieray run ... --seed 42

# Adjust refractive index (affects arrival_time calculation)
# The constant is in tracer.py: N_WATER_DEFAULT = 1.34

# Load LAPPD positions from a STEP CAD manifest
python -m annieray run ... --step F10091903_-.step --lappd-model annie

# Use a pre-built detector YAML instead of building from arrays
python -m annieray run ... --detector-config detectors.yaml

# Offset the PMT Z positions (useful if CSV and GDML differ)
python -m annieray run ... --z-offset -500.0
```

## How the Ray Tracing Works

### Pipeline overview

```
     GDML file          PMT CSV / STEP CAD
         |                     |
    gdml_parser.py       pmt_loader.py / step_parser.py
         |                     |
         v                     v
    build_geometry() ──────> Geometry dataclass
                                  |
                    ┌─────────────┴──────────────┐
                    v                             v
              trace_rays()                 trace_cherenkov()
                    |                             |
                    └─────────────┬───────────────┘
                                  v
                          trace_kernel()
                     (Taichi GPU kernel, lines 599-851
                      of tracer.py)
                                  |
                     ┌────────────┴────────────┐
                     v                         v
                hits ndarray (N, 13)    expand to (N, 15)
                     |                   with arrival_time
                     v                   + wavelength
                write_hits()
                     |
                     v
               hits.parquet
```

### The Taichi kernel (`trace_kernel`)

Each GPU thread processes one photon. For each photon the kernel:

1. **Normalises** the direction vector.
2. **Scans all mesh triangles** (structure) via Möller–Trumbore
   intersection (`_ray_triangle_intersect`), tracking the closest hit.
3. **Scans all PMTs** via sphere intersection (`_ray_sphere_intersect`).
   A hemisphere check (`dot(hit-centre, direction) > 0`) rejects back-face
   hits. On hit it computes local coordinates (`_pmt_local_coords`):
   polar angle θ from the PMT direction and azimuthal angle φ.
4. **Scans all default LAPPDs** via oriented-rectangle intersection
   (`_ray_rectangle_intersect`). On hit computes strip-aligned local
   coordinates (`_lappd_local_coords`): position along (u) and across (v)
   the photocathode strips.
5. **Finds the ANNIE LAPPD housing** (if present) via oriented-box slab
   intersection (`_ray_box_intersect`). Side/back faces absorb the photon;
   the front (+Z) face passes through to a separate photocathode rectangle.
6. **Finds the tank wall** via infinite-cylinder intersection
   (`_ray_tank_intersect`), clamped to the tank Z extent.
7. **Writes the nearest hit** to the output hit array: hit flag, path
   length, position, normal, component ID, detector index/system,
   and local coordinates.

All intersection functions live in `tracer.py` (lines 229–413).

### Detector registry

The `Geometry` object carries a `detectors: list[DetectorInfo]` built by
`build_detector_registry()` in `detectors.py`. Each `DetectorInfo` records
the stable ID (WCSim TubeIDs 332–463 for PMTs, 1000+ for default LAPPDs,
2000+ for ANNIE LAPPDs), system label, position, direction, panel number,
PMT type, and radius. The kernel writes `detector_index` (position in the
registry list) and `detector_system` (0=PMT, 1=default LAPPD, 2=ANNIE LAPPD)
into the hit array so downstream analysis can map hits back to hardware.

## Adding a Light-Emission Model

Photon generation is handled in `cli.py` and dispatched by the `--mode` flag.

### For a new emission distribution

Edit `_generate_*` functions in `cli.py` (lines 85–135) or create a new
module. Each function returns `(origins, directions)` — two `(N, 3)`
float32 numpy arrays in detector coordinates (mm). Then call
`trace_rays(origins, directions, geometry)` to run the GPU kernel.

Currently implemented:

- **`_generate_uniform`** (lines 85–118): rejection-samples positions
  uniformly inside a cylinder 90 % of the tank radius, with isotropic
  directions. Good for flat efficiency scans.

- **`_generate_cherenkov`** (lines 121–135): delegates to
  `cherenkov.generate_cherenkov_photons()`. Places all origins at a
  muon vertex and emits directions uniformly within a Cherenkov cone
  (cone angle 0.73 rad ≈ 42° for n=1.34). Random azimuth around the
  muon direction, random polar angle uniformly from 0 to the Cherenkov
  angle. Currently a single vertex — the muon is not propagated along
  its track.

To add e.g. electron-scintillation light, create:

```python
# cli.py (or new module)
def _generate_scintillation(geometry, n, rng, ...):
    origins = np.zeros((n, 3), dtype=np.float32)   # vertex positions
    directions = np.zeros((n, 3), dtype=np.float32) # unit directions
    # ... your generation logic ...
    return origins, directions
```

Then add a `--mode scintillation` branch in `run_command()`.

### Where to add wavelength-dependent physics

- **Wavelength per photon**: the `wavelength` argument is passed to
  `trace_cherenkov()` and stamped into hits[:, 14]. If you generate
  photons at multiple wavelengths, modify `_generate_cherenkov` to
  return a wavelength array and thread it through.

- **Cherenkov angle scaling**: the Cherenkov angle depends on wavelength
  via the refractive index. Currently `CHERENKOV_ANGLE = 0.73` is a
  constant in `cherenkov.py:5`. Replace it with a function like
  `cherenkov_angle(wavelength_nm)` that interpolates n(λ).

- **Refractive index**: `N_WATER_DEFAULT = 1.34` in `tracer.py:49`.
  The arrival-time calculation in `trace_cherenkov()` uses it via
  `c_in_water = C_MM_NS / n_water`. For wavelength-dependent timing,
  pass the per-photon n(λ) to that computation.

## Adding Scattering and Absorption

The current kernel does **no scattering or absorption** — photons travel
in straight lines from origin to the nearest surface. To add these:

### Rayleigh / Mie scattering in water

Would go in the kernel loop after the direction normalisation (line 634
of `tracer.py`). The conceptual approach:

1. Compute the scattering mean free path from water properties.
2. Step the photon along its direction in small increments.
3. At each step, roll against the scattering probability. On scatter,
   rotate the direction vector according to the appropriate phase
   function (Rayleigh: (1 + cos²θ); Mie: Henyey–Greenstein).

This requires a random number generator accessible inside the Taichi
kernel (`ti.random()` provides a per-thread RNG).

### Photon absorption

Would be implemented as:

- **Volume absorption**: at each scattering step, apply a survival
  probability `exp(-dx / absorption_length)` and discard (kill flag)
  if the photon is absorbed.

- **Surface absorption**: already partially implemented — photons hitting
  the housing side/back walls are absorbed (line 775: `best_hit = CID_NO_HIT`).
  For PMTs, the hemisphere check (`dot_fwd > 0` at line 704) already
  rejects back-face hits. A quantum-efficiency stage could further
  reject front-face hits by wavelength.

### Bulk scattering in acrylic windows (LAPPD housing)

The ANNIE LAPPD housing has an acrylic window (front face). To model
scattering there, extend the housing intersection logic in the kernel
to compute entry/exit points through the window slab and apply a
scattering/absorption model in between.

## Code Map

| File                    | Role                                           |
|-------------------------|------------------------------------------------|
| `cli.py`                | CLI parser, photon generation, main dispatch   |
| `tracer.py`             | Geometry dataclass, build_geometry, Taichi kernel, all intersection functions |
| `cherenkov.py`          | Cherenkov photon generation (single vertex)    |
| `detectors.py`          | DetectorInfo, build_detector_registry, YAML I/O |
| `output.py`             | Parquet writer, hit schema                     |
| `gdml_parser.py`        | Parse GDML mesh into vertex/triangle arrays    |
| `pmt_loader.py`         | Parse PMTPositions_Scan.txt into PMT arrays    |
| `step_parser.py`        | Parse STEP CAD into component manifest         |
| `lappd_model.py`        | Kandemir ANNIE LAPPD housing geometry builder  |
| `viz_server.py`         | Interactive Three.js viewer (aiohttp)          |
| `viz_lappd_server.py`   | Standalone LAPPD module viewer                 |
| `_version.py`           | Package version                                |
