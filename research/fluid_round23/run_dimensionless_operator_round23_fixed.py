from __future__ import annotations

import traceback

import run_dimensionless_operator_round23 as experiment

_ORIGINAL = experiment.r22.available_parents
_PARENT_BY_FACTOR = {
    str(identity["factor"]): str(identity["primitive_factor"])
    for identity in experiment.FROZEN_IDENTITIES
}


def available_parents_with_primitive(panel, factor: str):
    parents = list(_ORIGINAL(panel, factor))
    primitive = _PARENT_BY_FACTOR.get(str(factor))
    if primitive is not None and primitive in panel.columns and primitive not in parents:
        parents.append(primitive)
    return parents


experiment.r22.available_parents = available_parents_with_primitive

if __name__ == "__main__":
    try:
        experiment.main()
    except Exception as exc:  # noqa: BLE001
        failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        experiment.write_json(experiment.OUT / "FAILURE.json", failure)
        raise
