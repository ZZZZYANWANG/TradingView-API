# State-dependent non-Markovian fluid closure — Round20

## Decision

No candidate passed the predeclared joint validation gate. The 2025–2026 historical holdout and the second asset universe therefore remained closed, and no identity is eligible for the Trade factor catalog.

- Raw primitives: 16
- Smoothed candidate states: 48
- Development tests: 672
- Frozen family identities: 4
- Validation passes: 0
- Holdout opened: 0
- Replication opened: 0
- Confirmed identities: 0

The final evidence was produced by TimeSafe Actions run `44516352808`; artifact SHA-256: `b051df7d23b61a30f168a71b1852bece6d1708bcc6f7d9a5e71808169f422650`.

## Mechanism findings

### Lead-switch response

`lead_switch_response_ewm72`, 120 hours, was the strongest economic candidate. It conditions a causal distributed-lag shear response on whether spot historically leads perpetual or the reverse.

- Validation RankIC: 0.04285
- Residual RankIC: 0.03583
- Maximum parent correlation: 0.04616
- 5 bp annual log return: 17.42%
- 2023 / 2024 RankIC: 0.05686 / 0.02883
- Ridge MSE gain: -0.7767%
- Holm p: 0.1428

It has useful ranking and portfolio information but does not improve the frozen strong reader and does not survive family selection correction. It is not an independent alpha.

### Wave-action source phase

`wave_action_source_phase_ewm72`, 120 hours, retains positive direct and residual cross-sectional information and a positive Ridge loss increment, but has negative time-series IC and negative 5 bp return.

- Validation RankIC: 0.03207
- Residual RankIC: 0.04182
- Ridge MSE gain: +0.2105%
- TSIC: -0.03906
- 5 bp annual log return: -4.23%
- Holm p: 0.2613

It is better interpreted as a conditional amplitude or risk-state component, not a directional trading factor.

### Reflection hysteresis

`reflection_hysteresis_ewm72`, 120 hours, shows positive cross-sectional RankIC and low parent correlation, but has negative model increment and materially negative cost-adjusted return.

### Source-memory centroid loading

`source_memory_centroid_loading_ewm24`, 72 hours, was strongest in development but reversed in validation and in 2024. This rejects a fixed interpretation that the location of source-memory mass has a stable unconditional return sign.

## Structural conclusion

Conditioning the memory kernel on propagation direction improves the economic behavior of the response, but it does not create a regime-invariant unconditional alpha. The evidence supports modelling the transition operator

`market regime -> memory kernel -> response sign and amplitude`

rather than adding further static fluid-factor identities.

## Repository boundary

The code and complete failed-comparison evidence remain on the isolated research branch. Trade is intentionally unchanged because no declared use passed admission.
