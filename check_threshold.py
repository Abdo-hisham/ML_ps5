import mlflow
import os
import sys


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))

with open("model_info.txt", "r", encoding="utf-8") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

mlflow.set_tracking_uri(TRACKING_URI)

client = mlflow.tracking.MlflowClient()
try:
    run = client.get_run(run_id)
except Exception as exc:
    print(f"FAILED: Could not find run {run_id}: {exc}")
    sys.exit(1)

accuracy = run.data.metrics.get("accuracy", 0)
print(f"Model Accuracy: {accuracy}")

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy} is below threshold {THRESHOLD}")
    sys.exit(1) 
else:
    print(f"PASSED: Accuracy {accuracy} meets threshold {THRESHOLD}")
    sys.exit(0)  