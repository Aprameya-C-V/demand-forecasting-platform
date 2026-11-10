# Data card: NYC TLC yellow taxi

## Source

| | |
|---|---|
| Publisher | New York City Taxi and Limousine Commission |
| Dataset | Yellow Taxi Trip Records, monthly parquet |
| URL | `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet` |
| Zone lookup | `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv` |
| Licence | public domain, published by NYC Open Data |
| Cost | none, no authentication |

Each monthly file is a flat record of trips: pickup and dropoff timestamps, zone ids, distance,
fare components, payment type and passenger count. About 3 million rows per month, 60 MB parquet.

## What this project uses

Only three columns survive aggregation:

| Column | Use |
|---|---|
| `tpep_pickup_datetime` | floored to a 30 minute slot |
| `PULocationID` | pickup zone, filtered to 1 to 263 |
| count of rows | target: trips per zone per slot |

Everything else is dropped at ingestion. The aggregated panel stores counts only, so no row in this
repository can be traced back to a trip, a driver or a passenger.

## Panel definition

The panel is dense: every zone in 1 to 263 for every 30 minute slot in the window, with zero where
the TLC data has no pickups. Density is required because the model uses lags; a missing row would
make a lag silently skip a day. Validation enforces it.

Months in scope: 2026-05 through 2026-09 (October 2026 was not published at build time). 1.93M
rows before the 7 day lookback drop, 1.84M after.

## Known issues in the raw data

- **Out of range zones.** A few records carry zone ids of 264 or 265 (unknown / N/A). They are
  filtered out, not mapped.
- **Timestamps outside the file's month.** Some records are dated in the previous or next month.
  They are filtered by the half open month window so a trip is counted once.
- **Zero trip slots.** Real, not missing: late night zones genuinely see no pickups. The densify
  step is what makes this explicit.
- **Reporting lag.** The TLC publishes a month days to weeks after it ends, and occasionally
  revises a file. The manifest records the SHA-256 of every download so a revision is visible.

## Volume and shape

| | |
|---|---|
| Zones | 263 |
| Slot length | 30 minutes, 48 per day |
| Days in panel | 153 |
| Rows | 1,931,472 before lookback drop |
| Storage | 1.5 MB parquet after aggregation |

## Privacy

No identifiers, no coordinates finer than a TLC zone, no timestamps finer than 30 minutes in any
artefact that is committed. Raw trip files are git-ignored and never published.

## Reproducing

`forecast fetch` downloads with a 120 second timeout and rejects anything under 1 KB, which catches
the error page the CDN returns for unpublished months. `forecast panel` rebuilds the panel from
whatever is in `data/raw`. Both are idempotent and cached.
