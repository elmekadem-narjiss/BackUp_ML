import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import warnings
import itertools

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train_sarimax_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    with mlflow.start_run():
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Prepare data
        train_data = data['value'][:-24]
        test_data = data['value'][-24:]
        
        # Define model
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)
        
        # Log parameters
        mlflow.log_param("order", order)
        mlflow.log_param("seasonal_order", seasonal_order)
        
        # Make predictions
        predictions = fitted_model.forecast(steps=24)
        
        # Calculate metrics
        mse = mean_squared_error(test_data, predictions)
        mae = mean_absolute_error(test_data, predictions)
        r2 = r2_score(test_data, predictions)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(
            fitted_model,
            "sarimax_model"
        )
        
        logger.info("SARIMAX model trained and logged to MLflow")
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "model": fitted_model
        }

if __name__ == "__main__":
    # Example data
    data = pd.DataFrame({
        "value": np.random.rand(100)
    })
    result = train_sarimax_model(data)
    print(result)
