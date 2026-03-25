import os
import sys
from pathlib import Path

import mlflow


THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))
MODEL_INFO_PATH = Path(os.getenv("MODEL_INFO_PATH", "model_info.txt"))


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

    mlflow.set_tracking_uri(tracking_uri)

    try:
        run = mlflow.get_run(run_id)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: Failed to fetch run '{run_id}' from MLflow: {exc}")
        return 1

    metrics = run.data.metrics
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        print(f"ERROR: Run '{run_id}' has no 'accuracy' metric.")
        return 1

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD:.4f}")

    if accuracy < THRESHOLD:
        print("ERROR: Accuracy is below threshold. Deployment blocked.")
        return 1

    print("Accuracy meets threshold. Deployment can proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
