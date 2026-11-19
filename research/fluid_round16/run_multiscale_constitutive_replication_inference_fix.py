from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_multiscale_constitutive_inference_fix as inference  # noqa: E402
import run_multiscale_constitutive_replication as replication  # noqa: E402

OUT = Path("research_outputs/fluid_multiscale_constitutive_round16_replication_inference_fix")
OUT.mkdir(parents=True, exist_ok=True)

replication.OUT = OUT
replication.base.OUT = OUT
replication.base.moving_block_bootstrap = inference.nonoverlap_issue_block_bootstrap

if __name__ == "__main__":
    replication.main()
