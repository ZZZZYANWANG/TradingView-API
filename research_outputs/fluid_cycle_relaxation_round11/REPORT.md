# Cross-Cycle Relaxation Memory Test

The relaxation timescale is selected only from 2021–2024. The factor remains one identity in the cross-cycle-unwind information family.

## Timescale selection

|   span | factor                        |   development_cs_ic |   development_sharpe_2bp |   validation_cs_ic |   validation_residual_cs_ic |   validation_ridge_gain |   validation_sharpe_2bp |   validation_turnover |   validation_bootstrap_low |   positive_validation_years |   selection_score |
|-------:|:------------------------------|--------------------:|-------------------------:|-------------------:|----------------------------:|------------------------:|------------------------:|----------------------:|---------------------------:|----------------------------:|------------------:|
|     48 | cross_cycle_relaxation_ewm48  |         0.0188544   |                -0.26245  |          0.0519344 |                   0.0605525 |            -0.000232543 |               0.323405  |               1.361   |               -0.00276419  |                           1 |         0.0982615 |
|     24 | cross_cycle_relaxation_ewm24  |         0.0117668   |                -0.131046 |          0.0506925 |                   0.0569773 |            -0.000130889 |               0.485376  |               1.36599 |                0.00691894  |                           1 |         0.0958089 |
|     72 | cross_cycle_relaxation_ewm72  |         0.00590848  |                -0.376831 |          0.0505419 |                   0.0522731 |            -9.15885e-06 |               0.17192   |               1.34633 |               -0.0174695   |                           1 |         0.0919175 |
|     12 | cross_cycle_relaxation_ewm12  |        -0.018719    |                -0.327264 |          0.0427894 |                   0.0479828 |             0.000303629 |               0.43617   |               1.38152 |                0.000449722 |                           1 |         0.0848121 |
|    120 | cross_cycle_relaxation_ewm120 |        -0.013247    |                -0.415466 |          0.0400421 |                   0.04437   |             5.63366e-05 |               0.0236559 |               1.27733 |               -0.0216083   |                           1 |         0.0753642 |
|    240 | cross_cycle_relaxation_ewm240 |         0.000293542 |                -0.346692 |          0.0354885 |                   0.0287144 |            -1.30066e-05 |              -0.0955848 |               1.10598 |               -0.0190859   |                           1 |         0.0576297 |

## 2025 confirmation and 2026 shadow

| factor                       | period    |     cs_ic |   positive_share |   n_times |   bootstrap_low |   bootstrap_high |   p_nonpositive |   ridge_gain |   residual_cs_ic |   ts_ic_diagnostic |   ts_positive_share |   max_baseline_corr | max_corr_feature   |   net_sharpe_2bp |   annual_log_return_2bp |   avg_turnover | years                         |
|:-----------------------------|:----------|----------:|-----------------:|----------:|----------------:|-----------------:|----------------:|-------------:|-----------------:|-------------------:|--------------------:|--------------------:|:-------------------|-----------------:|------------------------:|---------------:|:------------------------------|
| cross_cycle_relaxation_ewm48 | 2025-2025 | 0.0291284 |         0.547945 |        73 |      -0.028165  |        0.0675147 |           0.235 | -0.00084278  |        0.0311606 |         -0.0241601 |            0.461538 |           0.0541368 | s_qz               |         0.424554 |               0.0579608 |        1.37889 | {"2025": 0.0579608059692974}  |
| cross_cycle_relaxation_ewm48 | 2026-2026 | 0.0600733 |         0.633333 |        60 |       0.0227083 |        0.12913   |           0     |  0.000795619 |        0.0577839 |          0.052972  |            0.769231 |           0.104255  | basis_z            |         2.34728  |               0.322355  |        1.38203 | {"2026": 0.32235517866908814} |

Final pass: **False**