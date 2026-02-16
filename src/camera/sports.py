"""Sport registry for calibration configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SportSpec:
    """Sport-specific geometry and semantic settings."""

    name: str
    court_size: Tuple[float, float]  # (length, width) in canonical court units
    num_regions: int


SPORT_SPECS: Dict[str, SportSpec] = {
    "basketball": SportSpec(
        name="basketball",
        # Using center-origin metric convention:
        # x in [-14.351, 14.351], y in [-7.6454, 7.6454]
        court_size=(28.702, 15.2908),
        num_regions=3,
    ),
    # Add soccer here later without refactoring pipeline code.
    # "soccer": SportSpec(name="soccer", court_size=(120.0, 80.0), num_regions=4),
}


def supported_sports() -> list[str]:
    return sorted(SPORT_SPECS.keys())


def get_sport_spec(sport: str) -> SportSpec:
    key = sport.lower()
    if key not in SPORT_SPECS:
        raise ValueError(
            f"Unsupported sport '{sport}'. Available: {', '.join(supported_sports())}"
        )
    return SPORT_SPECS[key]
