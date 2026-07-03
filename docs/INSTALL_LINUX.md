# Linux Installation (CPU, no GPU required)

This project is GPU-accelerated via [Taichi](https://taichi-lang.org/), but
Taichi also runs on CPU with no changes needed — all our launch code already
defaults to `ti.cpu`.  The interactive visualiser needs a browser but no
special graphics hardware.

## Prerequisites

- **Python ≥ 3.11** — check with `python3 --version`.  If older, install
  via your package manager (e.g. `apt install python3.11 python3.11-venv`
  on Debian/Ubuntu) or from [python.org](https://python.org).
- **git** — `apt install git` / `dnf install git`
- **C++ toolchain** (for Taichi's JIT compiler):
  - Debian/Ubuntu: `apt install build-essential cmake`
  - Fedora: `dnf install gcc-c++ cmake`
  - Arch: `pacman -S base-devel cmake`

## 1. Clone and initialise submodules

```bash
git clone https://github.com/wetstein/ANNIEraytracing.git
cd ANNIEraytracing
git submodule update --init
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Upgrade pip: `pip install --upgrade pip`

## 3. Install system packages (required by lxml)

Debian / Ubuntu:

```bash
sudo apt install libxml2-dev libxslt1-dev
```

Fedora:

```bash
sudo dnf install libxml2-devel libxslt-devel
```

These are only needed if you don't already have them — lxml's pip wheel may
already bundle them on your platform.

## 4. Install Python dependencies

**x86_64 (Intel/AMD) — simple:**

```bash
pip install -e .
```

This installs everything from `pyproject.toml`, including cadquery, which
distributes a manylinux wheel on x86_64.

**aarch64 (ARM, e.g. Raspberry Pi 4/5, Ampere) — cadquery needs conda:**

CadQuery does not publish a pip wheel for Linux ARM.  Use conda:

```bash
# Install miniforge (no NVIDIA GPU needed → no CUDA)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
# (follow the prompts, then restart your shell)

# Create environment with cadquery from conda-forge
conda create -n annie python=3.12
conda activate annie
conda install -c conda-forge cadquery pyarrow lxml pyyaml numpy
pip install taichi
```

Then install the package itself (still inside the conda env):

```bash
cd ANNIEraytracing
pip install -e . --no-deps
```

(`--no-deps` avoids re-resolving dependencies already satisfied by conda.)

**Alternative for ARM: skip cadquery entirely.** Most batch workflows use
`--pmt-csv` and never need cadquery.  Just `pip install -e .` will fail on
cadquery, so install the other deps by hand:

```bash
pip install taichi numpy pyarrow lxml pyyaml
pip install -e . --no-deps
```

## 5. Verify the installation

```bash
python -c "import annieray; print('OK')"
```

You should see `OK` with no errors.  If cadquery is missing, the package
will still import — cadquery is only used when `--step` or `--manifest`
are passed on the command line.

## 6. Run a test simulation

```bash
python -m annieray batch --events 10 --photons-per-cm 50
```

This runs without a GPU and writes results to `results/`.  Expected output:

```
Wrote results/photon_hits.parquet (NNNN photon rows)
Wrote results/muon_truth.parquet (10 muon truth rows)
Done.
```

### With PMT CSV and surfboard LAPPDs

```bash
# Get the PMT positions file (ask the collaboration)
# Then:
python -m annieray batch \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 --lappd-model annie \
    --events 100 --photons-per-cm 150
```

## 7. Interactive visualiser

The 3D viewer runs as a local web server — it serves a Three.js page to
your browser.  No GPU compute is needed on the server, just CPU ray tracing.

```bash
python -m annieray viz-server \
    --pmt-csv PMTPositions_Scan.txt \
    --port 8080
```

Open `http://localhost:8080` in any browser on the same machine (or another
machine on the same network).  Chrome, Firefox, and Edge all support
WebGL for the Three.js rendering.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `taichi.lang.exception.TaichiCompilationError: ...` | Taichi JIT needs C++ tools | `apt install build-essential cmake` |
| `ModuleNotFoundError: No module named 'cadquery'` | ARM Linux + no conda | Install via conda-forge (section 4) or use `--pmt-csv` workflow (doesn't need cadquery) |
| `Cannot open self /usr/bin/python3` | System Python too old | Install Python 3.11+ from deadsnakes PPA or python.org |
| Performance warning from Taichi | No GPU detected | Normal — CPU mode is expected; ~2-5× slower than GPU but still usable |
| `pyarrow.lib.ArrowInvalid: ...` | Corrupt parquet from killed job | Re-run with `--events` smaller or longer timeout |
