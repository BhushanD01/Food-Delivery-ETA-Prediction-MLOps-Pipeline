import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

def promote_staging_to_production():
    # 1. Initialize DagsHub and MLflow tracking environment
    print("Initializing DagsHub and MLflow client...")
    dagshub.init(
        repo_owner='BhushanD01', 
        repo_name='Food-Delivery-ETA-Prediction-MLOps-Pipeline', 
        mlflow=True
    )
    
    mlflow.set_tracking_uri("https://dagshub.com/BhushanD01/Food-Delivery-ETA-Prediction-MLOps-Pipeline.mlflow")
    client = MlflowClient()

    # 2. Load the active model name from your pipeline run information
    try:
        with open("run_information.json") as f:
            run_info = json.load(f)
            model_name = run_info["model_name"]
            print(f"Target model identified from registry: '{model_name}'")
    except Exception as e:
        print(f"Error loading run_information.json: {e}")
        exit(1)

    # 3. Fetch the latest model version currently sitting in 'Staging'
    print(f"Searching for the latest version of '{model_name}' in Staging...")
    staging_versions = client.get_latest_versions(name=model_name, stages=["Staging"])
    
    if not staging_versions:
        print(f"Error: No model version found in 'Staging' for model '{model_name}'.")
        exit(1)
        
    latest_staging_version = staging_versions[0].version
    print(f"Found Staging Version: {latest_staging_version}")

    # 4. Transition the staging model to Production
    print(f"Promoting version {latest_staging_version} to PRODUCTION...")
    client.transition_model_version_stage(
        name=model_name,
        version=latest_staging_version,
        stage="Production",
        archive_existing_versions=True  # Automatically archives the old production model
    )
    
    print(f"Success! Version {latest_staging_version} of '{model_name}' is now live in Production.")

if __name__ == "__main__":
    promote_staging_to_production()