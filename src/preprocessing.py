# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.config import PROCESSED_PATH, SEED, TEST_SIZE, TARGET

def preprocess_data(df):
    """
    Preprocesses the data by handling missing values, scaling features,
    and splitting into training and testing sets.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    # For simplicity, we'll fill missing values with the mean.
    # In a real-world scenario, more sophisticated methods might be needed.
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for later use in the app
    joblib.dump(scaler, os.path.join(PROCESSED_PATH, 'scaler.pkl'))

    # Save the processed data
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(os.path.join(PROCESSED_PATH, 'X_train.csv'), index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(os.path.join(PROCESSED_PATH, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_PATH, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_PATH, 'y_test.csv'), index=False)

    return X_train_scaled, X_test_scaled, y_train, y_test