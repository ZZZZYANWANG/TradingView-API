from __future__ import annotations

import json
import traceback

import run_state_dependent_memory_round20 as experiment

# Round18's reusable validation helper writes through its module-level OUT.
# Bind it to the Round20 evidence directory before any validation gate opens.
experiment.r18.OUT = experiment.OUT

if __name__ == "__main__":
    try:
        experiment.main()
    except Exception as exc:  # noqa: BLE001
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        experiment.write_json(experiment.OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
