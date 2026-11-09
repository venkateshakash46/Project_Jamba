
import json
import mlflow


mlflow_tracking_uri = (
    "sqlite:///D:/Arise/python/Mlops/mlflow.db"
)

mlflow.set_tracking_uri(mlflow_tracking_uri)


experiment_name = "ai21labs/Jamba-tiny-dev"

mlflow.set_experiment(experiment_name)


with open("D:\Arise\python\Mlops\LLMops\DataHouse\evaluation_results.json", "r") as f:
    results = json.load(f)


with mlflow.start_run(
    run_name="Jamba-tiny-dev_Run"
) as run:


    run_id = run.info.run_id
    experiment_id = run.info.experiment_id


    mlflow.log_param(
        "model_name",
        results["model_name"]
    )

    mlflow.log_param(
        "train_samples",
        results["train_samples"]
    )

    mlflow.log_param(
        "test_samples",
        results["test_samples"]
    )

    # Log training metrics
    for metric_name, value in results["train_metrics"].items():
        mlflow.log_metric(
            f"train_{metric_name}",
            value
        )


    for metric_name, value in results["test_metrics"].items():
        mlflow.log_metric(
            f"test_{metric_name}",
            value
        )


    mlflow.log_artifact(
        "D:\Arise\python\Mlops\LLMops\DataHouse\evaluation_results.json"
    )


print("Evaluation results logged successfully!")
print(f"Experiment Name: {experiment_name}")
print(f"Experiment ID: {experiment_id}")
print(f"Unique Run ID: {run_id}")
print(f"Tracking URI: {mlflow_tracking_uri}")
