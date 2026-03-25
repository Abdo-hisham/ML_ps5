import mlflow
import os
import sys
from pathlib import Path
from typing import Optional


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))


def _tracking_path_from_uri(tracking_uri: str) -> Path:
    if tracking_uri.startswith("file:"):
        return Path(tracking_uri.replace("file:", "", 1)).resolve()
    return Path(tracking_uri).resolve()


def _read_accuracy_from_mlruns(run_id: str, tracking_uri: str) -> Optional[float]:
    tracking_path = _tracking_path_from_uri(tracking_uri)
    candidate_roots = [tracking_path, Path.cwd(), Path.cwd() / "mlruns"]

    seen = set()
    for root in candidate_roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)

        for metrics_file in root.glob(f"*/{run_id}/metrics/accuracy"):
            try:
                lines = [ln.strip() for ln in metrics_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if not lines:
                    continue
                parts = lines[-1].split()
                if len(parts) >= 2:
                    return float(parts[1])
            except Exception:
                continue

        for metrics_file in root.rglob("accuracy"):
            try:
                if metrics_file.parent.name != "metrics":
                    continue
                if metrics_file.parent.parent.name != run_id:
                    continue
                lines = [ln.strip() for ln in metrics_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if not lines:
                    continue
                parts = lines[-1].split()
                if len(parts) >= 2:
                    return float(parts[1])
            except Exception:
                continue
    return None

with open("model_info.txt", "r", encoding="utf-8") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

mlflow.set_tracking_uri(TRACKING_URI)

client = mlflow.tracking.MlflowClient()
accuracy = None
try:
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")
except Exception as exc:
    print(f"INFO: API lookup failed, trying filesystem fallback: {exc}")
    accuracy = _read_accuracy_from_mlruns(run_id, TRACKING_URI)

if accuracy is None:
    accuracy = _read_accuracy_from_mlruns(run_id, TRACKING_URI)

if accuracy is None:
    print(f"FAILED: Could not find accuracy for run {run_id}")
    sys.exit(1)

print(f"Model Accuracy: {accuracy}")

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy} is below threshold {THRESHOLD}")
    sys.exit(1) 
else:
    print(f"PASSED: Accuracy {accuracy} meets threshold {THRESHOLD}")
    sys.exit(0)  