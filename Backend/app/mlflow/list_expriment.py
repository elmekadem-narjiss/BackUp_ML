from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

experiments = client.search_experiments()

for exp in experiments:
    print(f"[{exp.experiment_id}] {exp.name} (status: {exp.lifecycle_stage})")
