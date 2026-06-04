"""Load PMT positions from WCSim scan file, mapping to GDML coordinate frame.

Coordinate pipeline (scan → WCSim water-tank frame → structure rest frame):

  1. Scan file columns (cm):
       scan_x, scan_y, scan_z
     WCSim water-tank frame (mm):
       X_w  = scan_x * 10
       Y_w  = (168.1 - scan_z) * 10   (beam axis)
       Z_w  = (scan_y + 14.45) * 10   (vertical)
     where 168.1 cm = beam centre in scan-Z, 14.45 cm = vertical offset.

  2. The inner structure is placed in the water tank at:
       rotateZ(157.5°), G4ThreeVector(0, 0, -0.5*WCLength)
     with WCLength = 3960 mm (WCIDHeight).
     To get back to structure rest frame (GDML frame):
       P_s = Rz(-157.5°) * P_w + (0, 0, 0.5*WCLength)

  3. Direction vectors are generated from the panel_nr / PMT type
     (matching WCSim's per-face rotation matrices) rather than using
     the scan file direction columns, which are not used by WCSim.
     - Bottom PMTs (LUX)   →  (0, 0, 1)   upward
     - Top PMTs (ETEL)     →  (0, 0, -1)  downward
     - Barrel PMTs         →  radially inward in the XY plane
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# WCSim → tank-frame constants
# ---------------------------------------------------------------------------
# Beam centre Z-position in scan coordinates (cm)
BEAM_CENTER_Z = 168.1  # cm
# Vertical offset between scan and WCSim frames (cm)
VERTICAL_OFFSET = 14.45  # cm

# ---------------------------------------------------------------------------
# Structure placement in water tank
# ---------------------------------------------------------------------------
# The GDML structure is placed in the water tank with:
#   rotateZ(157.5°) at G4ThreeVector(0, 0, -0.5*WCLength)
# WCLength = WCIDHeight = 3960 mm
WC_LENGTH = 3960.0  # mm
HALF_WC_LENGTH = WC_LENGTH / 2.0  # 1980 mm

# Rotation angle to undo the structure placement
STRUCTURE_ROTATION_DEG = 157.5
_rot_rad = math.radians(-STRUCTURE_ROTATION_DEG)
_cos_r = math.cos(_rot_rad)
_sin_r = math.sin(_rot_rad)

# ---------------------------------------------------------------------------
# PMT radii by type (mm)
# ---------------------------------------------------------------------------
PMT_RADII = {
    "Hamamatsu": 101.6,  # 8-inch  (R5912)
    "Watchboy": 127.0,   # 10-inch (R7081)
    "Watchman": 127.0,   # 10-inch (R7081HQE)
    "LUX": 127.0,        # 10-inch (R7081)
    "ETEL": 139.7,       # 11-inch (D784KFLB)
}

# ---------------------------------------------------------------------------
# pmt_type → type name mapping (WCSim v7 collection order)
# ---------------------------------------------------------------------------
# WCTankCollectionNames for v7:
#   [0] R7081       → 10" LUX (panel 0) or Watchboy (panel 1-8)
#   [1] D784KFLB    → 11" ETEL (panel 9)
#   [2] R5912HQE    →  8" Hamamatsu (barrel)
#   [3] R7081HQE    → 10" Watchman (barrel)
# The scan file's pmt_type column indexes this vector.
# Panel number disambiguates pmt_type=0 (LUX vs Watchboy).
PMT_TYPE_NAMES: dict[tuple[int, int], str] = {}
# panel_nr=0 → LUX
for pt in (0,):
    PMT_TYPE_NAMES[(pt, 0)] = "LUX"
# panel_nr=9 → ETEL
for pt in (1,):
    PMT_TYPE_NAMES[(pt, 9)] = "ETEL"
# panel_nr=1-8, pmt_type=2 → Hamamatsu
for pn in range(1, 9):
    PMT_TYPE_NAMES[(2, pn)] = "Hamamatsu"
# panel_nr=1-8, pmt_type=3 → Watchman
for pn in range(1, 9):
    PMT_TYPE_NAMES[(3, pn)] = "Watchman"
# panel_nr=1-8, pmt_type=0 → Watchboy
for pn in range(1, 9):
    PMT_TYPE_NAMES[(0, pn)] = "Watchboy"


def _scan_to_tank(scan_x: float, scan_y: float, scan_z: float
                  ) -> tuple[float, float, float]:
    """Convert scan coordinates (cm) to WCSim water-tank frame (mm)."""
    x_w = scan_x * 10.0
    y_w = (BEAM_CENTER_Z - scan_z) * 10.0
    z_w = (scan_y + VERTICAL_OFFSET) * 10.0
    return x_w, y_w, z_w


def _tank_to_structure(x_w: float, y_w: float, z_w: float
                       ) -> tuple[float, float, float]:
    """Transform from water-tank frame to structure rest frame."""
    x_s = x_w * _cos_r - y_w * _sin_r
    y_s = x_w * _sin_r + y_w * _cos_r
    z_s = z_w + HALF_WC_LENGTH
    return x_s, y_s, z_s


def _ideal_direction(pmt_type: str, x_s: float, y_s: float
                     ) -> tuple[float, float, float]:
    """Return the ideal inward-pointing unit direction in structure frame.

    LUX (bottom) → upward (+Z)
    ETEL (top)   → downward (-Z)
    Barrel types → radially inward in XY plane
    """
    if pmt_type == "LUX":
        return 0.0, 0.0, 1.0
    if pmt_type == "ETEL":
        return 0.0, 0.0, -1.0
    # Barrel — radially inward
    r = math.sqrt(x_s * x_s + y_s * y_s)
    if r > 1e-6:
        return -x_s / r, -y_s / r, 0.0
    return 0.0, 0.0, 1.0


def load_pmts(scan_path: Path, z_offset: float = 0.0,
              bottom_rotation_deg: float = 0.0) -> dict:
    """Parse WCSim scan file and return PMT data in structure rest frame.

    Args:
        scan_path: Path to PMTPositions_Scan.txt.
        z_offset: Additional vertical shift (mm) applied after the
                  standard transform.  Use this to fine-tune alignment.
        bottom_rotation_deg: Extra rotation about Z for panel-0 (bottom)
                             PMTs only, in degrees.  Multiples of 22.5°
                             align with the 4-fold bottom structure.

    Returns dict with:
        centers: (N, 3) float32 — sphere centres in mm
        radii: (N,) float32 — per-PMT radius in mm
        types: list of str — PMT type name per PMT
        directions: (N, 3) float32 — inward-pointing face normal (unit)
        detector_nums: list of int — PMTID from scan file
    """
    data = np.loadtxt(scan_path, delimiter="\t")
    # Columns: TubeID, panel_nr, scan_x, scan_y, scan_z, dirx, diry, dirz, pmt_type
    tube_ids = data[:, 0].astype(int)
    panel_nrs = data[:, 1].astype(int)
    scan_xyz = data[:, 2:5]   # (N, 3) in cm
    pmt_types_code = data[:, 8].astype(int)

    n = len(data)
    centers = np.zeros((n, 3), dtype=np.float32)
    radii = np.zeros(n, dtype=np.float32)
    directions = np.zeros((n, 3), dtype=np.float32)
    type_names = []
    det_nums = []

    # Precompute bottom rotation
    _brot = math.radians(bottom_rotation_deg)
    _bcos = math.cos(_brot)
    _bsin = math.sin(_brot)

    for i in range(n):
        sx, sy, sz = scan_xyz[i]
        pn = panel_nrs[i]
        pt = pmt_types_code[i]

        # Scan → tank frame
        x_w, y_w, z_w = _scan_to_tank(sx, sy, sz)

        # Tank → structure rest frame
        x_s, y_s, z_s = _tank_to_structure(x_w, y_w, z_w)

        # Vertical offset
        z_s += z_offset

        # Extra Z-rotation for bottom panel (panel 0)
        if pn == 0 and bottom_rotation_deg != 0.0:
            xs = x_s
            x_s = xs * _bcos - y_s * _bsin
            y_s = xs * _bsin + y_s * _bcos

        # PMT type
        key = (pt, pn)
        ptype = PMT_TYPE_NAMES.get(key, "Unknown")
        radius = PMT_RADII.get(ptype, 127.0)

        # Idealized inward-pointing direction
        dx_s, dy_s, dz_s = _ideal_direction(ptype, x_s, y_s)

        centers[i] = [x_s, y_s, z_s]
        radii[i] = radius
        directions[i] = [dx_s, dy_s, dz_s]
        type_names.append(ptype)
        det_nums.append(int(tube_ids[i]))

    return {
        "centers": centers,
        "radii": radii,
        "types": type_names,
        "directions": directions,
        "detector_nums": det_nums,
        "panels": panel_nrs.tolist(),
    }
