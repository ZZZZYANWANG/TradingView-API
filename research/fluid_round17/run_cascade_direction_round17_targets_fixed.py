from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cascade_direction_round17 as experiment  # noqa: E402

# The inherited local feature builder creates targets from its module-level
# HORIZONS. Round17 intentionally adds 1h and 4h horizons, so bind the frozen
# Round17 horizon set before any data are loaded. No candidate definition,
# universe, orientation, span, selection rule or validation gate changes.
experiment.base.HORIZONS = list(experiment.HORIZONS)

if __name__ == "__main__":
    experiment.main()
