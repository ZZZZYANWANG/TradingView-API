from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent / "ashare_fluid_round10"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROUND10) not in sys.path:
    sys.path.insert(0, str(ROUND10))

import run_characteristic_network as research  # noqa: E402

research.source.MAX_CODES = 90
research.source.WORKERS = 6
research.OUT = Path("research_outputs/ashare_fluid_round15_reduced")
research.OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    research.main()
