from __future__ import annotations

import inspect

import run_hierarchical_constitutive_round25 as experiment

# The original 5,000-row guard was inherited from dense hourly readers. Round25
# issues non-overlapping 72h/120h observations. The partially pooled design is
# ridge-identified even when one-year crossfit has fewer rows than coefficients;
# asset deviations carry alpha=500 and the global block alpha=50. Keep a frozen,
# conservative floor of 60 observations and at least four observations per global
# constitutive basis term. Equations, universes, targets, penalties, horizons and
# admission gates are unchanged.
source = inspect.getsource(experiment.fit_model)
needle = "if int(valid.sum()) < 5000:"
replacement = "minimum_required = max(60, 4 * basis.shape[1])\n    if int(valid.sum()) < minimum_required:"
if needle not in source:
    raise RuntimeError("Round25 fit_model source contract changed")
namespace = experiment.__dict__
exec(source.replace(needle, replacement), namespace)
experiment.fit_model = namespace["fit_model"]


if __name__ == "__main__":
    experiment.main()
