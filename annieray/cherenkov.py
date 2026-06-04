"""Cherenkov photon generation for muon tracks in water."""

import numpy as np

CHERENKOV_ANGLE = 0.73  # radians, ~42 deg for n_water=1.34
DEFAULT_WAVELENGTH = 350.0  # nm


def generate_cherenkov_photons(
    muon_pos: tuple[float, float, float],
    muon_dir: tuple[float, float, float],
    n: int,
    cherenkov_angle: float = CHERENKOV_ANGLE,
    rng: np.random.Generator | None = None,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Cherenkov cone photons from a muon track.

    Args:
        muon_pos: Muon vertex (x, y, z) in mm.
        muon_dir: Muon direction (does not need to be unit).
        n: Number of photons to generate.
        cherenkov_angle: Cherenkov angle in radians.
        rng: NumPy random generator.
        wavelength: Photon wavelength in nm (default 350).

    Returns:
        (origins, directions) arrays, each (N, 3) float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    mdx, mdy, mdz = muon_dir
    m_len = np.sqrt(mdx * mdx + mdy * mdy + mdz * mdz)
    mdx /= m_len
    mdy /= m_len
    mdz /= m_len

    # Orthonormal basis (v, w) perpendicular to muon direction
    if abs(mdx) > 0.9:
        ux, uy, uz = 0.0, 1.0, 0.0
    else:
        ux, uy, uz = 1.0, 0.0, 0.0

    vx = uy * mdz - uz * mdy
    vy = uz * mdx - ux * mdz
    vz = ux * mdy - uy * mdx
    v_len = np.sqrt(vx * vx + vy * vy + vz * vz)
    vx /= v_len
    vy /= v_len
    vz /= v_len

    wx = mdy * vz - mdz * vy
    wy = mdz * vx - mdx * vz
    wz = mdx * vy - mdy * vx

    # Random azimuthal angles around the cone
    phi = rng.uniform(0, 2 * np.pi, n)
    # Sample angle uniformly from 0 to cherenkov_angle (filled cone)
    theta = rng.uniform(0, cherenkov_angle, n)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    origins = np.empty((n, 3), dtype=np.float32)
    directions = np.empty((n, 3), dtype=np.float32)

    origins[:, 0] = muon_pos[0]
    origins[:, 1] = muon_pos[1]
    origins[:, 2] = muon_pos[2]

    sp = np.sin(phi)
    cp = np.cos(phi)
    directions[:, 0] = mdx * cos_theta + (vx * cp + wx * sp) * sin_theta
    directions[:, 1] = mdy * cos_theta + (vy * cp + wy * sp) * sin_theta
    directions[:, 2] = mdz * cos_theta + (vz * cp + wz * sp) * sin_theta

    return origins, directions
