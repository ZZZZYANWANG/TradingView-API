from __future__ import annotations

import json
import traceback

import numpy as np

import run_regime_constitutive_round24_fixed as fixed

experiment = fixed.experiment


def safe_development_crossfit(panel, spec):
    """Run one-year cross-fit diagnostics without weakening the frozen two-year fit.

    The 72h/120h non-overlapping issuance clocks leave about 100 observations in a
    single development year after warm-up. These diagnostics are recorded when
    estimable, but they are not part of the admission gate and must not abort the
    two-year 2021-2022 frozen model, which retains the 300-observation minimum.
    """
    output = []
    for train_year, test_year in [(2021, 2022), (2022, 2021)]:
        try:
            model = fixed.fit_model(
                panel,
                str(spec["parent"]),
                int(spec["horizon"]),
                train_year,
                train_year,
            )
        except RuntimeError as exc:
            output.append(
                {
                    "factor": spec["factor"],
                    "train_year": train_year,
                    "test_year": test_year,
                    "status": "insufficient_nonoverlap_for_diagnostic",
                    "reason": str(exc),
                    "ts_ic": np.nan,
                    "positive_asset_share": np.nan,
                    "annual_log_return_5bp": np.nan,
                    "sharpe_5bp": np.nan,
                }
            )
            continue

        temporary = f"__crossfit_{spec['factor']}_{test_year}"
        experiment.apply_model(panel, model, temporary)
        frame = experiment.r22.issue_frame(
            panel,
            temporary,
            int(spec["horizon"]),
            test_year,
            test_year,
        )
        summary = experiment.r22.ts_summary(frame, temporary, int(spec["horizon"]))
        account = experiment.r23.execution_account(
            frame,
            temporary,
            int(spec["horizon"]),
            5.0,
        )
        panel.drop(columns=[temporary], inplace=True)
        output.append(
            {
                "factor": spec["factor"],
                "train_year": train_year,
                "test_year": test_year,
                "status": "estimated",
                "reason": "",
                "ts_ic": float(summary["ts_ic"]),
                "positive_asset_share": float(summary["positive_asset_share"]),
                "annual_log_return_5bp": float(account["annual_log_return"]),
                "sharpe_5bp": float(account["sharpe"]),
            }
        )
    return output


experiment.development_crossfit = safe_development_crossfit

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
