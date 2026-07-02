"""Generate a MuonStartsAndDirecs file with angular scan.

For each (x,z) position in the standard grid, creates 9 muon
directions: forward along +Y plus ±22.5° in the vertical (Y-Z)
and horizontal (X-Y) planes, including all 4 diagonal combos.

Output format: <x> <y> <z> <t0> <dx> <dy> <dz>

Usage:
    python scripts/generate_muon_grid.py > MuonStartsAndDirecs_angled.txt
"""

import itertools
import math

NX = 13
NZ = 13
X_MIN, X_MAX = -1200.0, 1200.0
Z_MIN, Z_MAX = 300.0, 2700.0
Y = 0.0
T0 = 0.0

ANGLES_DEG = [-22.5, 0.0, 22.5]


def direction(vert_deg: float, horiz_deg: float):
    v = math.radians(vert_deg)
    h = math.radians(horiz_deg)
    cv, sv = math.cos(v), math.sin(v)
    ch, sh = math.cos(h), math.sin(h)
    dx = cv * sh
    dy = cv * ch
    dz = sv
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dx / norm, dy / norm, dz / norm


def main():
    xs = [X_MIN + i * (X_MAX - X_MIN) / (NX - 1) for i in range(NX)]
    zs = [Z_MIN + i * (Z_MAX - Z_MIN) / (NZ - 1) for i in range(NZ)]

    for x, z in itertools.product(xs, zs):
        for vert, horiz in itertools.product(ANGLES_DEG, repeat=2):
            dx, dy, dz = direction(vert, horiz)
            print(f"{x:g} {Y:g} {z:g} {T0:g} {dx:g} {dy:g} {dz:g}")


if __name__ == "__main__":
    main()
