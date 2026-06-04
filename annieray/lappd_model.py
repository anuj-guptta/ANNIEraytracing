"""LAPPD housing model based on Kandemir's WCSim implementation.

Provides the "ANNIE" LAPPD model: a waterproof housing with an acrylic
window, air gap, and off-center photocathode, replacing the Default model's
bare rectangle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- Default model (bare photocathode rectangle) ----
DEFAULT_HALF_SIZE = 101.0  # mm — side = 202 mm square

# ---- ANNIE model (housed LAPPD) dimensions (mm) ----
# Housing outer box half-extents
HOUSING_HALF = (165.0, 215.0, 30.0)   # X × Y × Z (Z = radial direction)

# Photocathode half-sizes
PC_HALF = (95.75, 95.75)

# Photocathode centre in the housing local frame
PC_LOCAL = (0.0, -45.0, 3.5)   # (X, Y, Z) — offset -45 mm in Y

# Position correction ratio (shifts LAPPD radially inward)
CORRECTION = 0.965


@dataclass
class LAPPDHousing:
    """Oriented-box housing with internal photocathode."""

    centre: tuple[float, float, float]          # box centre (mm), after 0.93 correction
    axes: tuple[                                # orthonormal right-handed axes
        tuple[float, float, float],             # local X (tangential)
        tuple[float, float, float],             # local Y (vertical = +Z in structure frame)
        tuple[float, float, float],             # local Z (radially inward = front face)
    ]
    half: tuple[float, float, float] = HOUSING_HALF  # half-extents

    # Pre-computed world-frame photocathode
    pc_centre: tuple[float, float, float] | None = None
    pc_normal: tuple[float, float, float] | None = None
    pc_half: tuple[float, float] = PC_HALF


def build_housing(
    cad_centre: tuple[float, float, float],
    cad_normal: tuple[float, float, float],
    z_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> LAPPDHousing:
    """Build the housed LAPPD at a CAD candidate position.

    Parameters
    ----------
    cad_centre:
        (x, y, z) from the STEP CAD (mm).
    cad_normal:
        Radially inward unit normal at the CAD position.
    z_axis:
        Vertical direction in the structure frame (default Z-up).

    Returns
    -------
    LAPPDHousing with all fields filled.
    """
    cx, cy, cz = cad_centre
    nx, ny, nz = cad_normal

    # Apply 0.93 radial correction in XY only (Z unchanged)
    hx = cx * CORRECTION
    hy = cy * CORRECTION
    hz = cz

    # Local axes (right-handed)
    local_z = (nx, ny, nz)  # front face normal = radial inward

    # local_y follows z_axis (vertical)
    local_y = z_axis

    # local_x = cross(local_y, local_z), then re-orthogonalise
    lx = (
        local_y[1] * local_z[2] - local_y[2] * local_z[1],
        local_y[2] * local_z[0] - local_y[0] * local_z[2],
        local_y[0] * local_z[1] - local_y[1] * local_z[0],
    )
    ll = math.sqrt(lx[0]**2 + lx[1]**2 + lx[2]**2)
    if ll > 1e-12:
        local_x = (lx[0] / ll, lx[1] / ll, lx[2] / ll)
    else:
        local_x = (1.0, 0.0, 0.0)

    # Recompute local_y = cross(local_z, local_x) for orthogonality
    local_y = (
        local_z[1] * local_x[2] - local_z[2] * local_x[1],
        local_z[2] * local_x[0] - local_z[0] * local_x[2],
        local_z[0] * local_x[1] - local_z[1] * local_x[0],
    )

    # Photocathode world position
    plx, ply, plz = PC_LOCAL
    pc_world = (
        hx + plx * local_x[0] + ply * local_y[0] + plz * local_z[0],
        hy + plx * local_x[1] + ply * local_y[1] + plz * local_z[1],
        hz + plx * local_x[2] + ply * local_y[2] + plz * local_z[2],
    )

    return LAPPDHousing(
        centre=(hx, hy, hz),
        axes=(local_x, local_y, local_z),
        pc_centre=pc_world,
        pc_normal=local_z,
    )


def housing_to_arrays(housing: LAPPDHousing) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a LAPPDHousing into kernel arrays.

    Returns
    -------
    housing_data : ndarray (1, 16) float32
        [cx,cy,cz, ax_x,ax_y,ax_z, ay_x,ay_y,ay_z, az_x,az_y,az_z, hx,hy,hz, pad]
    annie_lappd_data : ndarray (1, 7) float32
        [pcx,pcy,pcz, pcnx,pcyn,pczn, pchalf] — same layout as lappd_data.
    """
    ax, ay, az = housing.axes
    hx, hy, hz = housing.half
    hd = np.array([
        housing.centre[0], housing.centre[1], housing.centre[2],
        ax[0], ax[1], ax[2],
        ay[0], ay[1], ay[2],
        az[0], az[1], az[2],
        hx, hy, hz,
        0.0,  # padding
    ], dtype=np.float32).reshape(1, 16)

    pc = np.array([
        housing.pc_centre[0], housing.pc_centre[1], housing.pc_centre[2],
        housing.pc_normal[0], housing.pc_normal[1], housing.pc_normal[2],
        housing.pc_half[0],
    ], dtype=np.float32).reshape(1, 7)

    return hd, pc
