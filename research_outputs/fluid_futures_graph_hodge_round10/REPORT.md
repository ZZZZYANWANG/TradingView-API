# Cross-Asset Futures Graph Hodge Test

## Development family winners

| factor                                | family                          |   horizon |   sign |   development_ts_ic |   development_positive_share |
|:--------------------------------------|:--------------------------------|----------:|-------:|--------------------:|-----------------------------:|
| cross_basis_pressure_release_ewm72    | cross_basis_pressure_release    |       120 |     -1 |           0.0224114 |                        0.625 |
| cross_cycle_trap_ewm12                | cross_cycle_trap                |        24 |      1 |           0.0628623 |                        1     |
| cross_cycle_unwind_ewm12              | cross_cycle_unwind              |       120 |      1 |           0.0327897 |                        0.75  |
| cross_hodge_consensus_ewm12           | cross_hodge_consensus           |        12 |      1 |           0.0579724 |                        1     |
| cross_pressure_basis_conversion_ewm72 | cross_pressure_basis_conversion |       120 |     -1 |           0.0773764 |                        0.625 |
| cross_shear_absorption_ewm12          | cross_shear_absorption          |        72 |      1 |           0.0305839 |                        0.625 |

## 2023-2024 validation

| factor                                |   horizon |   sign |       ts_ic |   ts_positive_share |       cs_ic |   cs_bootstrap_low |   ridge_gain |   residual_ts_ic |   max_baseline_corr | max_corr_feature     |   net_sharpe_2bp |   annual_log_return_2bp |   avg_turnover | years                                                        | family                          |   development_ts_ic |   selection_score | validation_gate   |
|:--------------------------------------|----------:|-------:|------------:|--------------------:|------------:|-------------------:|-------------:|-----------------:|--------------------:|:---------------------|-----------------:|------------------------:|---------------:|:-------------------------------------------------------------|:--------------------------------|--------------------:|------------------:|:------------------|
| cross_shear_absorption_ewm12          |        72 |      1 |  0.0377548  |               0.625 | -0.00175644 |         -0.0515247 | -3.22335e-05 |       0.0367193  |           0.0431846 | s_imb_z              |        -0.542088 |              -0.134505  |        1.4049  | {"2023": -0.02363218092251739, "2024": -0.24537682387636559} | cross_shear_absorption          |           0.0305839 |        0.0651816  | False             |
| cross_pressure_basis_conversion_ewm72 |       120 |     -1 |  0.0695319  |               0.625 |  0.0280496  |         -0.0419235 | -0.00988048  |       0.0770117  |           0.155105  | p_mom72              |         0.382077 |               0.110135  |        1.39465 | {"2023": -0.20305325422050216, "2024": 0.4233232786152787}   | cross_pressure_basis_conversion |           0.0773764 |        0.0334827  | False             |
| cross_cycle_trap_ewm12                |        24 |      1 |  0.0429651  |               1     |  0.0122793  |         -0.0146904 | -0.00160476  |       0.00547773 |           0.801057  | s_mom24              |        -0.743427 |              -0.178449  |        1.27471 | {"2023": -0.16163698833500623, "2024": -0.19521489037668574} | cross_cycle_trap                |           0.0628623 |        0.0290695  | False             |
| cross_hodge_consensus_ewm12           |        12 |      1 |  0.0241191  |               1     |  0.00737737 |         -0.0113079 | -0.00177989  |       0.00438172 |           0.680364  | s_mom24              |        -1.28734  |              -0.302576  |        1.16507 | {"2023": -0.4076934233574187, "2024": -0.19774583708325785}  | cross_hodge_consensus           |           0.0579724 |        0.00596358 | False             |
| cross_basis_pressure_release_ewm72    |       120 |     -1 |  0.00143542 |               0.5   |  0.00782779 |         -0.047142  | -2.77646e-05 |       0.00171594 |           0.159388  | cross_graph_pressure |         0.376542 |               0.0943766 |        1.34549 | {"2023": -0.19574767584424313, "2024": 0.3845008012756876}   | cross_basis_pressure_release    |           0.0224114 |        0.00366015 | False             |
| cross_cycle_unwind_ewm12              |       120 |      1 | -0.0159804  |               0.5   |  0.0189172  |         -0.0132216 |  0.000331428 |      -0.0145503  |           0.0686528 | s_rv_z               |         1.0788   |               0.244426  |        1.3879  | {"2023": 0.23307542938708342, "2024": 0.2557771284152939}    | cross_cycle_unwind              |           0.0327897 |       -0.02107    | False             |

Final pass: **False**