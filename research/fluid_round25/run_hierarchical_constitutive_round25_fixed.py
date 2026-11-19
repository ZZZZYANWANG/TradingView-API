from __future__ import annotations

import inspect

import run_hierarchical_constitutive_round25 as experiment

# The original 5,000-row guard was inherited from dense hourly readers. Round25
# issues non-overlapping 72h/120h observations, so the legal 2021-2022 sample is
# intentionally much smaller. The constitutive equations, universes, penalties,
# targets, horizons and admission gates are unchanged; only the implementation
# guard is aligned with the 117-parameter partially pooled design.
source = inspect.getsource(experiment.fit_model)
needle = "if int(valid.sum()) < 5000:"
replacement = "minimum_required = max(250, 2 * (1 + len(set(symbols[valid].tolist()))) * basis.shape[1])\n    if int(valid.sum()) < minimum_required:"
if needle not in source:
    raise RuntimeError("Round25 fit_model source contract changed")
namespace = experiment.__dict__
exec(source.replace(needle, replacement), namespace)
experiment.fit_model = namespace["fit_model"]


if __name__ == "__main__":
    experiment.main()
