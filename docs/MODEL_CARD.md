# Model card: demand forecasting

## Summary

| | |
|---|---|
| Task | Forecast taxi trips per TLC zone per 30 minute slot, one day ahead |
| Model | LightGBM gradient boosted trees, one point model and two quantile models |
| Version served | read from `GET /v1/model` or `reports/monitor.json`, never hard coded here |
| Owner | Aprameya C V, personal project |
| Licence | code: MIT |
| Last evaluation | 14 day holdout, 17 to 30 September 2026 |

## Intended use

Position supply: decide which zones need vehicles in the next day. The forecast is a planning
input, not a live dispatch signal, and it is not a demand measurement for billing.

**Not intended for:** individual trip pricing, driver compensation, capacity commitments, or any
use where a missed peak has safety consequences. The model covers one day and aggregate zones.

## Data

See [`DATA_CARD.md`](DATA_CARD.md). Training window: 8 May 2026 to 26 August 2026. Validation:
27 August to 16 September 2026. Test: 17 to 30 September 2026. The split is strictly time based;
no future information reaches the features.

## Features

| Feature | Definition | Available at prediction time |
|---|---|---|
| `zone_id` | TLC pickup zone, 1 to 263 | yes |
| `hour` | hour of day | yes |
| `dow` | day of week, Monday = 0 | yes |
| `slot` | 30 minute slot index, 0 to 47 | yes |
| `is_weekend` | Saturday or Sunday | yes |
| `lag_1d` | trips in the same slot yesterday | yes |
| `lag_7d` | trips in the same slot a week ago | yes |
| `roll_7d_slot` | mean of the same slot over the previous 7 days | yes |
| `roll_7d_zone` | mean demand across all slots for the zone over 7 days | yes |

No feature uses the target day, so the matrix for tomorrow is built entirely from realised history.

## Training

- Objective: Poisson for the point model, quantile loss at 0.1 and 0.9 for the band.
- Early stopping on the validation window, 50 rounds patience, up to 1,200 rounds.
- Split by time, never at random. Shuffling rows across days would leak tomorrow into training.
- Seed fixed at 7. Feature importance is gain based; `lag_7d` carries most of it.

## Evaluation

Held out window, 176,736 rows:

| Metric | Value | Notes |
|---|---|---|
| WAPE | 0.218 | total absolute error over total demand |
| MAE | 2.32 | trips per slot |
| RMSE | 6.92 | dominated by high volume zones |
| Bias | -2.5% | mild under-forecast |
| Coverage p10-p90 | 0.794 | nominal 0.80 |
| Max PSI | 0.017 | served window vs training reference |

WAPE by hour ranges from 0.176 at 18:00 to 0.444 at 04:00. Low volume hours are harder on a ratio
metric; absolute error there is small.

## Failure modes

- **Disruption.** Weather, strikes, major events and road closures are not modelled. During those
  days the model is expected to be wrong, and the monitor should be read as an alert, not a bug.
- **Level shift.** `roll_7d_zone` reacts within a week, so a sudden structural change (a new
  pickup restriction) takes days to absorb.
- **Sparse zones.** Zones with a handful of trips per slot show large percentage error and wide
  bands. Use the interval, not the point.
- **Stale panel.** The API is only as current as the last published month. It reports
  `data_through` on `/readyz` so this is visible.

## Ethical and governance notes

- No personal data: the panel is counts per zone and slot, derived from trip records with no
  identifiers.
- Zone level aggregates can still describe demand patterns in specific neighbourhoods. Do not
  present them as a proxy for the people who live there.
- Any deployment must keep the "planning input" framing visible to users.

## Maintenance

Retraining is scheduled monthly and gated. A challenger that does not beat production by 2% WAPE
stays registered but unserved. Drift is checked daily against a reference frozen at training time;
if the reference becomes stale, retrain and the new reference replaces it.
