# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-11

First production ready release.

### Added

- **Ingestion with provenance.** Monthly NYC TLC downloads with a SHA-256 manifest, size guard
  against error pages, and a cached re-run path.
- **Panel contract.** Aggregation to a dense zone x 30 minute slot panel, validated for density,
  nulls, duplicates, negatives and zone count before any model sees it.
- **Feature cube.** Lags and rolling demand computed by reshaping the dense panel, so training and
  serving share one code path.
- **Training.** Time based split, LightGBM Poisson point model plus p10/p90 quantile models, early
  stopping on a validation window, metadata and a frozen drift reference.
- **Registry and gate.** Versioned artefacts, `production` alias, and a relative improvement gate
  so a retrain cannot promote on noise.
- **Service.** FastAPI with liveness and readiness separation, model metadata, single zone and
  batch forecasts, Prometheus counters and histograms, JSON access logs, request IDs, a model
  version header and `Server-Timing`.
- **Monitoring.** WAPE, MAE, RMSE, bias, interval coverage and per feature PSI on replayed
  demand, with thresholds that fail a workflow and open a GitHub issue.
- **Dashboard.** Static monitoring page generated from the same JSON the workflows commit.
- **Automation.** CI with lint, format, strict types, coverage gate at 85% and a real pipeline
  smoke test; monthly retraining; daily monitoring; container build and publish to GHCR; CodeQL.
- **Operations.** Runbook with alert response, rollback, and the failure modes already seen.
  Model card and data card.
- **Load test script** and the measured figures in the README.

### Changed

- Batch endpoint raised from 50 to 263 zones to cover the whole feature store.
- Prediction logging keeps one buffered handle instead of reopening the file per request, which
  cut p50 latency from 39 ms to 3.3 ms at one client.
- Future slot lookup precomputes per zone indices instead of scanning the matrix per request.
- Service can run multiple uvicorn workers through `DF_WORKERS`.

### Fixed

- A forecast start beyond the last served slot returned an inverted window. The window is now
  `[forecast_start, forecast_end]` and out of range starts are a 422.
- A test fixture that stopped before the end of the month produced an empty holdout window and
  degenerate zero metrics. The fixture now covers the month, matching how the panel is built.

## [0.1.0] - 2026-11-10

Initial working pipeline: ingestion, panel, training, registry, service, monitoring, CI.
