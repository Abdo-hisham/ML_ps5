import os
import sys
from pathlib import Path
from typing import Optional

import mlflow


THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))
MODEL_INFO_PATH = Path(os.getenv("MODEL_INFO_PATH", "model_info.txt"))


def _tracking_path_from_uri(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        return Path(tracking_uri.replace("file:", "", 1)).resolve()
    return Path(tracking_uri).resolve()


def _read_accuracy_from_mlruns(run_id: str, tracking_uri: str) -> Optional[float]:
    tracking_path = _tracking_path_from_uri(tracking_uri)
    if not tracking_path.exists():
        return None

    for metrics_file in tracking_path.glob(f"*/{run_id}/metrics/accuracy"):
        try:
            lines = [ln.strip() for ln in metrics_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                continue
            parts = lines[-1].split()
            if len(parts) >= 2:
                return float(parts[1])
        except Exception:
            continue

    return None


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI is not set.")
        return 1

    if not MODEL_INFO_PATH.exists():
        print(f"ERROR: {MODEL_INFO_PATH} does not exist.")
        return 1

    run_id = MODEL_INFO_PATH.read_text(encoding="utf-8").strip()
    if not run_id:
        print("ERROR: model_info.txt is empty.")
        return 1

    print(f"Checking Run ID: {run_id}")

    mlflow.set_tracking_uri(tracking_uri)

    accuracy = None
    try:
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy")
    except Exception:
        # Fallback for artifact-restored file store when API lookup is flaky.
        accuracy = _read_accuracy_from_mlruns(run_id, tracking_uri)

    if accuracy is None:
        print(f"FAILED: Could not find accuracy for run {run_id}")
        return 1

    print(f"Accuracy: {accuracy}")
    if accuracy < THRESHOLD:
        print(f"FAILED: {accuracy} is below {THRESHOLD}")
        return 1

    print(f"PASSED: {accuracy} meets threshold {THRESHOLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
