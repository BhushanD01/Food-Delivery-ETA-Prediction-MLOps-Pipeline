import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

dagshub.init(repo_owner='BhushanD01', repo_name='Food-Delivery-ETA-Prediction-MLOps-Pipeline', mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/BhushanD01/Food-Delivery-ETA-Prediction-MLOps-Pipeline.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


if __name__ == "__main__":

    # root path
    root_path = Path(__file__).parent.parent.parent
    run_info_path = root_path / "run_information.json"
    run_info = load_model_information(run_info_path)

    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    artifact_path = run_info["artifact_path"] 

    model_registry_path = artifact_path
    
    try:
        model_version = mlflow.register_model(
            model_uri=model_registry_path,
            name=model_name
        )
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True
        )
        client.set_registered_model_alias(model_name, "production", model_version.version)
        logger.info(f"Successfully registered {model_name} version {model_version.version}")
    except Exception as e:
        print(f"Registration failed: {e}")
    