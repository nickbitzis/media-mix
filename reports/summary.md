# Study summary

Juniper Row (made up): 156 weeks (2023-01-02 to 2025-12-22), fit on 130, holdout 26.

Holdout MAPE: base 1.10%, calibrated 1.12%.

Lift test (search, 80 regions, weeks 93 to 98): iROAS 0.99 (90% CI 0.87 to 1.11), true 1.05.

## Recovered vs true ROI (training weeks, 90% block bootstrap intervals)

| Channel | True ROI | Model | 90% CI | Error | Calibrated | 90% CI | Error |
|---|---:|---:|---|---:|---:|---|---:|
| TV | 1.46 | 1.70 | 1.44 to 1.93 | +16% | 1.79 | 1.50 to 1.99 | +22% |
| Paid search | 1.11 | 1.65 | 0.80 to 1.88 | +48% | 0.94 | 0.92 to 0.95 | -15% |
| Paid social | 1.88 | 2.51 | 1.86 to 3.04 | +33% | 2.96 | 2.15 to 3.43 | +58% |
| Display | 0.79 | 0.32 | 0.00 to 0.87 | -59% | 0.22 | 0.00 to 0.74 | -72% |
| Podcast | 1.50 | 0.00 | 0.00 to 0.04 | -100% | 0.08 | 0.00 to 0.16 | -95% |
| Influencer | 2.40 | 2.90 | 2.54 to 3.23 | +21% | 2.93 | 2.55 to 3.21 | +22% |
| Email | 3.96 | 3.95 | 3.61 to 4.53 | -0% | 3.92 | 3.58 to 4.46 | -1% |

## Budget plan, $4,676,494 for the quarter (calibrated model)

| Channel | Current | Optimized | Optimized 90% range |
|---|---:|---:|---|
| TV | $2,181,173 | $2,286,007 | $2,211,204 to $2,347,375 |
| Paid search | $933,600 | $834,880 | $795,445 to $903,476 |
| Paid social | $597,282 | $541,895 | $439,746 to $605,945 |
| Display | $376,282 | $188,141 | $188,141 to $262,032 |
| Podcast | $233,828 | $116,914 | $116,914 to $116,914 |
| Influencer | $276,684 | $553,369 | $553,369 to $553,369 |
| Email | $77,644 | $155,288 | $155,288 to $155,288 |

Predicted media revenue lift +10.3% (90% range +8.8% to +11.8%), true lift +9.0%. Uncalibrated model's plan: predicted +9.8%, true +7.0%.

## Gate (tolerance 30%): PASS

- Influencer: recovered 2.93, true 2.40, error +21.9%: pass
- Email: recovered 3.92, true 3.96, error -1.0%: pass
- Paid social + display combined: recovered 1.73, true 1.39, error +24.4%: pass
