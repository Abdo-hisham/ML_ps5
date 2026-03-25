import os
from pathlib import Path

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_and_evaluate() -> float:
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )

    model = LogisticRegression(max_iter=300)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return float(accuracy_score(y_test, predictions))


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Default")
    mlflow.set_experiment(experiment_name)

    override = os.getenv("MOCK_ACCURACY")

    with mlflow.start_run() as run:
        if override is not None and override.strip() != "":
            accuracy = float(override)
            mlflow.log_param("accuracy_source", "override")
        else:
            accuracy = train_and_evaluate()
            mlflow.log_param("accuracy_source", "model")

        mlflow.log_param("model_name", "logreg_iris")
        mlflow.log_metric("accuracy", accuracy)

        Path("model_info.txt").write_text(run.info.run_id, encoding="utf-8")
        print(f"Run ID: {run.info.run_id}")
        print(f"Logged accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
