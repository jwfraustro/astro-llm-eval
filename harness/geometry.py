"""Small spherical-geometry helpers so the control box needs no astropy."""

from __future__ import annotations

import math


def angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in degrees (Vincenty form; stable at small angles)."""
    l1, b1, l2, b2 = map(math.radians, (ra1, dec1, ra2, dec2))
    dl = l2 - l1
    sin_b1, cos_b1 = math.sin(b1), math.cos(b1)
    sin_b2, cos_b2 = math.sin(b2), math.cos(b2)
    cos_dl, sin_dl = math.cos(dl), math.sin(dl)

    num = math.hypot(cos_b2 * sin_dl, cos_b1 * sin_b2 - sin_b1 * cos_b2 * cos_dl)
    den = sin_b1 * sin_b2 + cos_b1 * cos_b2 * cos_dl
    return math.degrees(math.atan2(num, den))
