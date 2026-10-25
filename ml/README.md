# Phase 3 — KNN traffic-volume prediction

scikit-learn K-Nearest-Neighbours regression that predicts traffic **volume**
for a `(road_id, day_of_week, hour)` bucket.

## Method
* **Features** (all known before the measurement — no leakage):
  * `road_id` → one-hot (so an arbitrary id is never treated as a numeric distance)
  * `hour`, `day_of_week` → cyclical `sin/cos` (23:00 sits next to 00:00)
  * numeric features standardised (KNN is distance-based)
* **Target**: `volume`. The target is never in the feature vector.
* **Split**: chronological by **replay order** — train on the first `1-test_frac`
  of rows, test on the final `test_frac`. The raw data has no absolute
  timestamp, so replay/stream order is the leakage-free notion of "time".
* **Baseline**: historical mean volume per `(road_id, hour)` from the training
  split only (global train mean for unseen buckets).
* **Metrics**: MAE, RMSE, R² for both KNN and baseline.

## Run
```bash
pip install -r requirements-dev.txt
python python/prepare_dataset.py
python ml/knn_predict.py --k 7 --weights distance --test-frac 0.2
```

Outputs to `data/models/`: `knn_predictions.csv`, `knn_metrics.json`,
`knn_model.joblib`.

## Measured result (full dataset, k=7, distance weights)
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| KNN | **6.66** | **9.35** | **0.849** |
| Baseline (per road/hour mean) | 14.44 | 18.34 | 0.418 |

KNN reduces MAE by **~54%** vs the baseline. See `docs/REPORT_PROJECT.md` for the
honest discussion of limitations (no calendar timestamp; cyclic time only).
