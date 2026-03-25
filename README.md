# MLflow Tracking

This project logs training runs to MLflow.

## View MLflow UI locally

Run these commands from the project folder:

```powershell
python -m pip install -r requirements.txt
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open this link in your browser:

http://127.0.0.1:5000

## Notes

- In GitHub Actions, `MLFLOW_TRACKING_URI` is set to `file:./mlruns`.
- That means runs are stored as files and uploaded as workflow artifacts.
- There is no public MLflow web link unless you deploy an MLflow server.
