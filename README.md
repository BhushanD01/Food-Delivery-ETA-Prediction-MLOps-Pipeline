# Food-Delivery-ETA-Prediction-MLOps-Pipeline
## 🚀 Project Overview

Predicting an accurate Estimated Time of Arrival (ETA) is a critical component of logistics and food delivery platforms to ensure customer satisfaction and driver dispatch optimization. A company wants to optimize delivery time predictions to improve customer experience by providing accurate estimated delivery times and to manage resources effectively. Accurate predictions of delivery time can also allow the business to:

1. Improve Delivery Efficiency: Identifying factors that slow down deliveries enables better resource allocation, such as more reliable scheduling for delivery personnel.
2. Enhance Customer Satisfaction: Reliable delivery ETAs can improve the customer experience by reducing wait-time uncertainty.
3. Optimize Operational Costs: If the model can predict the uncertainties like high demand, additional resources (more drivers) can be allocated.

## ⚒️ Tech Stack:

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![NumPy Badge](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=fff&style=for-the-badge)
![Pandas Badge](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=fff&style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white&style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4EAEAA?logo=python&logoColor=fff&style=for-the-badge)
![scikit-learn Badge](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=fff&style=for-the-badge)
![DVC Badge](https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=fff&style=for-the-badge)
![MLflow Badge](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=fff&style=for-the-badge)
![GitHub Actions Badge](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=fff&style=for-the-badge)
![Docker Badge](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff&style=for-the-badge)
![AWS](https://custom-icon-badges.demolab.com/badge/AWS-%23FF9900.svg?logo=aws&logoColor=white&style=for-the-badge)
<br><br>

## 📁 Project Structure
    ├── .github/
    │   └──workflows/    
    │       └──ci-cd.yaml        # CI/CD Workflow
    ├── data                     # data folder tracked by DVC
    │   └──raw  
    │   └──processed 
    │   └──interim 
    │   └──cleaned
    │   └──external  
    ├── models                   # models folder tracked by DVC
    ├── scripts
    │   └──data_clean_func.py    # data cleaning functions
    │   └──promote_model.py      # script to promote model to production
    │   └──sample_prediction.py  # prediction testing script
    │   └──test_performance.py   # performance testing script  
    ├── src
    │   └──data
    │       └──data_cleaning.py
    │       └──data_preparation.py 
    │   └──features
    │       └──data_preprocessing.py
    │   └──model
    │       └──model_train.py
    │       └──model_evaluation.py
    │       └──register_model.py
    │
    ├── Dockerfile
    ├── README.md
    ├── app.py                   # FastAPI application
    │
    ├── dvc.yaml                 # Pipeline execution blueprint
    │
    ├── params.yaml              # Centralized hyperparameters config
    │
    ├── requirements-dev.txt     # Development & testing dependencies
    │
    ├── requirements-docker.txt  # container-specific dependencies

## 💻 Local Setup

**Option A: Running via Docker Hub**
- The entire application layer is containerized and published to Docker Hub. If you just want to spin up the live FastAPI server instantly without downloading data or cloning the source files, pull and run the image directly:
```python
docker pull dock114/food_delivery_time_pred
```
**Option B: Clone the Repository & Initialize Environment**
 - Clone the repository
```python
https://github.com/BhushanD01/Food-Delivery-ETA-Prediction-MLOps-Pipeline.git
```
 - Upgrade pip and install development/testing packages
```python
pip install --upgrade pip
pip install -r requirements-dev.txt
```
 - Pull artifacts via DVC
```python
dvc pull
```
