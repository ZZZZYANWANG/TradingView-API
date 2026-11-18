from __future__ import annotations

import refine_hodge_residual as study

# Frozen focused replication of the single mechanism nearest admission in the
# preceding rolling-graph experiment.  No additional mechanism families are
# searched here.
study.RAW_FAMILIES = ["curl_gradient_conversion"]
study.CONFIGS = [(3, 0.05, 1.0), (5, 0.05, 0.5), (5, 0.05, 1.0)]
study.SMOOTH = [32, 64]
study.HORIZONS = [10]

if __name__ == "__main__":
    study.main()
