from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent / "ashare_fluid_round10"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROUND10) not in sys.path:
    sys.path.insert(0, str(ROUND10))

import run_characteristic_network as research  # noqa: E402

spec = importlib.util.spec_from_file_location("ashare_round10_core", ROUND10 / "run.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import round10 core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

original_universe = core.universe
core.universe = lambda n_each=250: original_universe(150)
research.source.load_baostock = core.load
research.OUT = Path("research_outputs/ashare_fluid_round15_fast")
research.OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    research.main()
