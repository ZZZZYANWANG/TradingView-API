# Coupled Spot–Perpetual Daily Fluid Validation

## Selection

| family                | factor                |   horizon |   sign |   development_ts_ic |   development_cs_ic |   selection_score |
|:----------------------|:----------------------|----------:|-------:|--------------------:|--------------------:|------------------:|
| interfacial_energy    | energy_flux           |        20 |      1 |           0.229269  |          -0.0484694 |          0.412802 |
| kelvin_helmholtz      | kh_onset              |        10 |     -1 |           0.105023  |           0.0471861 |          0.247975 |
| interfacial_vorticity | vorticity_10          |        10 |      1 |           0.0839623 |           0.12522   |          0.27249  |
| continuity_pressure   | pressure_drive        |        10 |      1 |           0.158487  |           0.0186147 |          0.329583 |
| sweeping_impedance    | interfacial_impedance |        10 |     -1 |           0.0880231 |           0.0298701 |          0.199622 |
| interfacial_geometry  | interfacial_torsion   |        20 |      1 |           0.0847833 |           0.0114638 |          0.17759  |

## Validation

| family                | factor                |   horizon |   sign |   development_ts_ic |   development_cs_ic |   selection_score | validation_pass   |   validation_ts_ic |   validation_ts_positive_share |   validation_cs_ic |   validation_boot_low |   validation_ridge_gain |   validation_residual_ts_ic |   validation_residual_cs_ic |   validation_max_baseline_corr | validation_max_corr_feature   |   validation_net_sharpe_5bp |   validation_net_annual_log_return_5bp | validation_years                                            |
|:----------------------|:----------------------|----------:|-------:|--------------------:|--------------------:|------------------:|:------------------|-------------------:|-------------------------------:|-------------------:|----------------------:|------------------------:|----------------------------:|----------------------------:|-------------------------------:|:------------------------------|----------------------------:|---------------------------------------:|:------------------------------------------------------------|
| interfacial_energy    | energy_flux           |        20 |      1 |           0.229269  |          -0.0484694 |          0.412802 | False             |         0.0206259  |                          0.5   |        0.1139      |             0.020592  |                     nan |                  0.0763691  |                   0.0534106 |                      0.646955  | activity_gap_z                |                    1.68295  |                               0.444682 | {"2023": 0.45210764319884483, "2024": 0.4494400193118557}   |
| kelvin_helmholtz      | kh_onset              |        10 |     -1 |           0.105023  |           0.0471861 |          0.247975 | False             |         0.0805889  |                          0.875 |        0.0302445   |            -0.0383124 |                     nan |                 -0.0938837  |                  -0.016731  |                      0.770611  | shear_z                       |                    1.20284  |                               0.316974 | {"2023": 0.23823557899185938, "2024": 0.40439688412732017}  |
| interfacial_vorticity | vorticity_10          |        10 |      1 |           0.0839623 |           0.12522   |          0.27249  | False             |        -0.0847279  |                          0.5   |       -0.0579151   |            -0.136157  |                     nan |                 -0.109422   |                  -0.0694981 |                      0.0911474 | p_mom20                       |                    0.639329 |                               0.195023 | {"2023": 0.38330753046489896, "2024": 0.012080818395874703} |
| continuity_pressure   | pressure_drive        |        10 |      1 |           0.158487  |           0.0186147 |          0.329583 | False             |         0.00352092 |                          0.375 |        0.0395753   |            -0.0376448 |                     nan |                  0.0266827  |                   0.0199485 |                      0.112686  | basis_velocity_z              |                    1.13532  |                               0.260737 | {"2023": 0.3639017712050461, "2024": 0.1647158030623264}    |
| sweeping_impedance    | interfacial_impedance |        10 |     -1 |           0.0880231 |           0.0298701 |          0.199622 | False             |         0.0692929  |                          0.75  |        0.0408623   |            -0.0318855 |                     nan |                  0.00928175 |                   0.0189833 |                      0.981899  | shear_z                       |                    1.60485  |                               0.427084 | {"2023": 0.2894383385909359, "2024": 0.5764297930422941}    |
| interfacial_geometry  | interfacial_torsion   |        20 |      1 |           0.0847833 |           0.0114638 |          0.17759  | False             |        -0.0306721  |                          0.5   |        0.000643501 |            -0.124839  |                     nan |                 -0.0082385  |                  -0.0212355 |                      0.165873  | activity_gap_z                |                    0.691714 |                               0.227163 | {"2023": 0.20843988800722288, "2024": 0.25210908519222647}  |

## Conditional holdout

No validation passer; holdout remained sealed.

Promoted factors: 0