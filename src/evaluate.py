# src/evaluate.py

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import MODEL_PATH, PROCESSED_PATH

def evaluate_model():
    """
    Evaluates the trained model on the test set and logs the metrics.
    """
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(os.path.join(PROCESSED_PATH, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_PATH, 'y_test.csv')).values.ravel()

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")

    # Error distribution plot
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    
    # Create output directory for plots
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.show()

if __name__ == '__main__':
    evaluate_model()