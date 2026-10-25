"""Phase 3 — K-Nearest-Neighbours traffic-volume prediction.

Problem definition
------------------
* **Target**    : ``volume`` (traffic intensity). Chosen over "speed" because the
                  raw dataset's measurement behaves like volume (rises at commute
                  peaks) and is the only numeric measure present.
* **Granularity**: one prediction per (``road_id``, ``day_of_week``, ``hour``).
* **Features**  : only values known *before* the measurement is taken —
                  ``road_id`` (one-hot), ``hour`` and ``day_of_week``
                  (cyclical sin/cos so 23:00 is "near" 00:00 and the metric
                  never treats an arbitrary id as a numeric distance).
* **Horizon**   : predicts the volume for a given road at a given day/hour
                  bucket.
* **Evaluation**: chronological holdout by **replay order**. The raw data has
                  no absolute timestamp, so "time" = the order rows are
                  streamed (= file order). We train on the first ``1-test_frac``
                  of rows and test on the final ``test_frac`` — no future row
                  ever informs a past prediction, and the target is never in
                  the feature vector.
* **Baseline**  : historical mean volume per (road_id, hour) learned from the
                  training split only (global train mean for unseen buckets).

Honest limitations are documented in docs/REPORT_PROJECT.md: with only cyclic
time (no calendar date) a pure replay-order split is the strongest leakage-free
evaluation available on this dataset.

Run:
    python ml/knn_predict.py --k 7 --weights distance --test-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config  # noqa: E402
from common.logging_setup import get_logger  # noqa: E402
from common.schema import load_canonical  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

LOG = get_logger("knn_predict")

RANDOM_SEED = 42


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    n_test: int

    def as_dict(self) -> dict:
        return asdict(self)


def _one_hot_encoder() -> OneHotEncoder:
    # Compatible across sklearn versions (sparse_output is >=1.2).
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - old sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature frame (cyclical time + road id)."""
    out = pd.DataFrame(index=df.index)
    out["road_id"] = df["road_id"].astype(str)
    out["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    return out


def make_pipeline(k: int, weights: str) -> Pipeline:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")
    numeric = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    pre = ColumnTransformer(
        transformers=[
            ("road", _one_hot_encoder(), ["road_id"]),
            ("time", StandardScaler(), numeric),
        ]
    )
    return Pipeline(
        [("pre", pre), ("knn", KNeighborsRegressor(n_neighbors=k, weights=weights))]
    )


def chronological_split(df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split preserving row (replay) order: first = train, last = test."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0,1), got {test_frac}")
    n_test = max(1, int(round(len(df) * test_frac)))
    n_train = len(df) - n_test
    if n_train < 1:
        raise ValueError("Not enough rows to form a training split.")
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def _metrics(actual: np.ndarray, pred: np.ndarray) -> Metrics:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return Metrics(mae=mae, rmse=rmse, r2=r2, n_test=len(actual))


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Historical mean volume per (road_id, hour) from the training split."""
    table = train.groupby(["road_id", "hour"])["volume"].mean()
    global_mean = float(train["volume"].mean())
    keys = list(zip(test["road_id"], test["hour"]))
    return np.array([table.get(k, global_mean) for k in keys], dtype=float)


def train_and_evaluate(
    df: pd.DataFrame, *, k: int = 7, weights: str = "distance", test_frac: float = 0.2
) -> dict:
    """Train KNN, evaluate against a replay-order holdout and a baseline."""
    if len(df) < max(10, k + 1):
        raise ValueError(
            f"Need at least {max(10, k + 1)} rows to train KNN with k={k}; got {len(df)}."
        )

    train, test = chronological_split(df, test_frac)
    LOG.info("Split: train=%d test=%d (replay-order holdout)", len(train), len(test))

    pipe = make_pipeline(k, weights)
    pipe.fit(build_features(train), train["volume"].to_numpy())

    knn_pred = pipe.predict(build_features(test))
    base_pred = baseline_predict(train, test)
    actual = test["volume"].to_numpy()

    knn_metrics = _metrics(actual, knn_pred)
    base_metrics = _metrics(actual, base_pred)
    LOG.info("KNN      MAE=%.3f RMSE=%.3f R2=%.3f", knn_metrics.mae, knn_metrics.rmse, knn_metrics.r2)
    LOG.info("Baseline MAE=%.3f RMSE=%.3f R2=%.3f", base_metrics.mae, base_metrics.rmse, base_metrics.r2)

    predictions = test[["road_id", "day_of_week", "hour"]].copy()
    predictions["actual_volume"] = actual
    predictions["predicted_volume"] = np.round(knn_pred, 3)
    predictions["baseline_volume"] = np.round(base_pred, 3)
    predictions["target_unit"] = "volume_index"

    return {
        "pipeline": pipe,
        "predictions": predictions,
        "knn": knn_metrics,
        "baseline": base_metrics,
        "params": {"k": k, "weights": weights, "test_frac": test_frac, "seed": RANDOM_SEED},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="KNN traffic-volume prediction")
    parser.add_argument("--k", type=int, default=int(__import__("os").environ.get("KNN_K", 7)))
    parser.add_argument("--weights", default="distance", choices=["uniform", "distance"])
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--input", default=str(config.CANONICAL_CSV))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        LOG.error("Input not found: %s. Run python/prepare_dataset.py first.", input_path)
        return 1

    df = load_canonical(input_path)
    LOG.info("Loaded %d canonical rows across %d roads", len(df), df["road_id"].nunique())

    result = train_and_evaluate(df, k=args.k, weights=args.weights, test_frac=args.test_frac)

    config.ensure_dir(config.MODELS_DIR)
    pred_path = config.MODELS_DIR / "knn_predictions.csv"
    result["predictions"].to_csv(pred_path, index=False)
    LOG.info("Wrote predictions -> %s", pred_path)

    metrics_path = config.MODELS_DIR / "knn_metrics.json"
    improvement = (
        (result["baseline"].mae - result["knn"].mae) / result["baseline"].mae * 100
        if result["baseline"].mae
        else float("nan")
    )
    metrics_path.write_text(
        json.dumps(
            {
                "knn": result["knn"].as_dict(),
                "baseline": result["baseline"].as_dict(),
                "mae_improvement_pct_vs_baseline": round(improvement, 2),
                "params": result["params"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info("Wrote metrics -> %s (MAE improvement vs baseline: %.1f%%)", metrics_path, improvement)

    try:
        import joblib

        model_path = config.MODELS_DIR / "knn_model.joblib"
        joblib.dump(result["pipeline"], model_path)
        LOG.info("Saved model -> %s", model_path)
    except Exception as exc:  # noqa: BLE001 - model persistence is best-effort
        LOG.warning("Model not persisted (%s)", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
