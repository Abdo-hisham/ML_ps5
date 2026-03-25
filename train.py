import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("assignment5")

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # Allow forcing pass/fail scenarios from workflow_dispatch.
    mock_accuracy = os.getenv("MOCK_ACCURACY", "").strip()
    if mock_accuracy:
        accuracy = float(mock_accuracy)

    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(clf, name="model")

        print(f"Accuracy: {accuracy}")
        print(f"Run ID: {run.info.run_id}")
        with open("model_info.txt", "w", encoding="utf-8") as f:
            f.write(run.info.run_id)

    print("model_info.txt saved successfully!")


if __name__ == "__main__":
    main()