# src/data_loader.py

import pandas as pd
import os
from src.config import DATA_PATH

def load_data(file_name):
    """
    Loads a CSV file from the raw data directory.

    Args:
        file_name (str): The name of the file to load.

    Returns:
        pandas.DataFrame: The loaded dataframe.
    """
    csv_path = os.path.join(DATA_PATH, file_name)
    return pd.read_csv(csv_path)