# Interfacial Energy Flux — Frozen Holdout Confirmation

Holdout pass: **False**

The formula, sign and 20-day horizon were frozen after 2021-2022 development and 2023-2024 validation.
The 2025-2026 segment is used here as the first confirmation segment for this identity.

## Validation

```json
{
  "daily_cs_ic": 0.029262944769393468,
  "daily_cs_positive_share": 0.5417236662106704,
  "daily_cs_n": 731,
  "daily_cs_bootstrap": {
    "n": 731,
    "mean": 0.029262944769393468,
    "low": -0.0244397404937109,
    "high": 0.07830271643541138,
    "p_nonpositive": 0.139
  },
  "ts": {
    "mean": 0.007605969930393402,
    "median": -0.0011787576495307962,
    "positive_share": 0.5,
    "n_symbols": 8,
    "by_symbol": {
      "ADAUSDT": -0.025432185314132126,
      "BNBUSDT": -0.07372133881675859,
      "BTCUSDT": -0.03213425646738101,
      "DOGEUSDT": 0.0764017764012757,
      "ETHUSDT": 0.06290851123898397,
      "LINKUSDT": 0.023074670015070534,
      "SOLUSDT": 0.1524310606323593,
      "XRPUSDT": -0.12268047824627058
    }
  },
  "offset_positive_share": 0.6,
  "offset_median_cs_ic": 0.017651580151580158,
  "leave_one_out_min_cs_ic": -0.0026220160361076815,
  "leave_one_out_positive_share": 0.875
}
```

## Holdout

```json
{
  "daily_cs_ic": -0.06734781421613661,
  "daily_cs_positive_share": 0.4367283950617284,
  "daily_cs_n": 648,
  "daily_cs_bootstrap": {
    "n": 648,
    "mean": -0.06734781421613661,
    "low": -0.1297332075522172,
    "high": -0.000983954946628061,
    "p_nonpositive": 0.9775
  },
  "ts": {
    "mean": 0.04179575586752679,
    "median": 0.07050037692237274,
    "positive_share": 0.75,
    "n_symbols": 8,
    "by_symbol": {
      "ADAUSDT": 0.09909925405170736,
      "BNBUSDT": 0.13203472653869533,
      "BTCUSDT": -0.10824828062142394,
      "DOGEUSDT": 0.01124267588784394,
      "ETHUSDT": 0.05035040776117526,
      "LINKUSDT": -0.03661057170081873,
      "SOLUSDT": 0.09584748893946493,
      "XRPUSDT": 0.0906503460835702
    }
  },
  "offset_positive_share": 0.15,
  "offset_median_cs_ic": -0.04941152597402598,
  "leave_one_out_min_cs_ic": -0.09291389879061444,
  "leave_one_out_positive_share": 0.0
}
```

## Holdout residual

```json
{
  "daily_cs_ic": -0.0047766019988242205,
  "daily_cs_positive_share": 0.48302469135802467,
  "daily_cs_n": 648,
  "daily_cs_bootstrap": {
    "n": 648,
    "mean": -0.0047766019988242205,
    "low": -0.05636849647266315,
    "high": 0.039722957084068186,
    "p_nonpositive": 0.569
  },
  "ts": {
    "mean": 0.0008795186031059568,
    "median": -0.004393011931062386,
    "positive_share": 0.5,
    "n_symbols": 8,
    "by_symbol": {
      "ADAUSDT": -0.02314882070206514,
      "BNBUSDT": -0.036007390066541814,
      "BTCUSDT": 0.054790784573615366,
      "DOGEUSDT": -0.00981955179948526,
      "ETHUSDT": 0.0671888860399064,
      "LINKUSDT": -0.07510694048299162,
      "SOLUSDT": 0.001033527937360489,
      "XRPUSDT": 0.028105653325049232
    }
  },
  "offset_positive_share": 0.45,
  "offset_median_cs_ic": -0.004464285714285711,
  "leave_one_out_min_cs_ic": -0.024415784832451503,
  "leave_one_out_positive_share": 0.5
}
```

## Holdout model increment

```json
{
  "n_train": 10238,
  "n_test": 5184,
  "mse_base": 0.024328684293951495,
  "mse_plus": 0.02432571529094317,
  "relative_gain": 0.00012203713823278282,
  "daily_loss_diff_mean": 2.96900300832594e-06,
  "daily_loss_diff_bootstrap": {
    "n": 648,
    "mean": 2.96900300832594e-06,
    "low": -1.7153702050941165e-05,
    "high": 2.25695456461587e-05,
    "p_nonpositive": 0.367
  }
}
```

## Holdout portfolios

|   cost_bp |   n_days |   annual_log_return |   gross_annual_log_return |   annual_volatility |    sharpe |   avg_daily_turnover | years                                                        |
|----------:|---------:|--------------------:|--------------------------:|--------------------:|----------:|---------------------:|:-------------------------------------------------------------|
|         0 |      668 |           -0.125644 |                 -0.125644 |            0.432776 | -0.290321 |             0.122016 | {'2025': -0.11131532763405022, '2026': -0.11862977604764012} |
|         2 |      668 |           -0.134551 |                 -0.125644 |            0.432761 | -0.310912 |             0.122016 | {'2025': -0.11905904563713055, '2026': -0.12718734956398767} |
|         5 |      668 |           -0.147912 |                 -0.125644 |            0.43274  | -0.341802 |             0.122016 | {'2025': -0.13067462264175128, '2026': -0.14002370983850887} |
|        10 |      668 |           -0.170179 |                 -0.125644 |            0.432705 | -0.393292 |             0.122016 | {'2025': -0.1500339176494523, '2026': -0.16141764362937763}  |