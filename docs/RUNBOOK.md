# Runbook

Operational guide for the demand forecasting service. Everything here is what an on-call engineer
does when a workflow goes red or a forecast looks wrong.

## 1. Where things run

| Component | Host | Notes |
|---|---|---|
| Pipelines | GitHub Actions | free tier, public repository |
| Model artefacts | repository `models/` (git-ignored) and workflow artefacts | 30 day retention |
| Forecast service | any container host | `Dockerfile` at the root, no external state |
| Monitoring output | `reports/` and `docs/dashboard/data.json` | committed so the dashboard updates |

There is no database. State is the model registry directory plus the committed dashboard payload,
which keeps the cost at zero and makes rollback a one line change.

## 2. Alert response

`monitor.yml` fails and opens an issue titled `Monitor breach: <date>` when any of these trip:

| Alert | Meaning | First action |
|---|---|---|
| `WAPE above threshold` | accuracy decay on realised demand | check if a holiday or disruption is in the window, then retrain |
| `bias outside ±10%` | systematic over or under forecasting | inspect the level features (`roll_7d_zone`) for a step change |
| `feature drift, max PSI` | input distribution moved | look at the feature named in the report |
| `below the minimum rows` | the window is too small to judge | check that the panel covers the window |

Order of operations:

1. Read `reports/monitor.json` from the workflow run artefacts.
2. Compare `metrics.by_hour` with the previous run. A single hour moving is noise; the whole
   profile moving is real.
3. If the label window contains a known event (storm, holiday, marathon), annotate the issue and do
   not retrain on it. Retraining would bake the anomaly into the weights.
4. Otherwise retrain via `train.yml` and let the gate decide.

## 3. Retraining

```bash
gh workflow run train.yml
gh run watch
```

The workflow fetches the newest published months, rebuilds the panel, trains, registers the
challenger, then promotes only if WAPE improves by `registry.min_improvement` (2%).

The job summary prints:

- challenger version, WAPE, MAE, coverage
- champion version and the metric it was compared on
- the decision and the reason string from `should_promote`

If the challenger loses, it stays registered. Nothing else happens. That is the intended outcome,
not a failure.

## 4. Rollback

Pinning production to an earlier version is one command, because aliases are what the service
reads:

```bash
uv run python - <<'PY'
from pathlib import Path
from demand_forecast.registry import ModelRegistry

registry = ModelRegistry(Path("models"))
registry.set_alias("production", "<previous-version>")
PY
```

Commit the alias file, redeploy, and confirm with `curl /v1/model`. The version string in the
response must match.

## 5. Service checks

```bash
curl -s localhost:8000/readyz      # status, model version, data_through
curl -s localhost:8000/v1/model    # metrics the model was promoted on
curl -s localhost:8000/metrics | grep forecast_
```

A `degraded` readiness means no alias or no feature store. The response body carries the reason.

Cold start loads the panel and the three boosters, roughly two seconds for this dataset. If
`readyz` is degraded after startup, check that `DF_MODEL_DIR` and `DF_PANEL_PATH` point at real
paths.

## 6. Data refresh

The panel ends at the last published TLC month. When a new month appears:

1. add it to `data.months` in `configs/base.yaml`
2. add the month to the `months` list used by the train workflow matrix
3. run `train.yml`

The API horizon extends automatically, because the predictor derives `forecast_start` from the last
slot in the panel.

## 7. Failure modes already seen

| Symptom | Cause | Fix |
|---|---|---|
| `download too small, probably an error page` | TLC has not published that month yet | use the newest complete month |
| Panel validation fails on zone count | TLC changed the zone set | update `zone_min` / `zone_max` and re-check the zone lookup |
| `Library not loaded: libomp.dylib` | macOS without OpenMP | `brew install libomp` |
| Forecast returns 422 | `start` outside the forecastable window | call `/v1/model` and use the returned window |

## 8. On-call checklist before a release

- [ ] CI green on `main`
- [ ] `forecast pipeline` runs end to end locally
- [ ] monitor reports `passed: true`
- [ ] dashboard renders with the new `data.json`
- [ ] model card updated with the new metrics
