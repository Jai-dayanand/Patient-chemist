# src/config.py

# Define paths relative to the project root
DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
MODEL_PATH = "models/model.pkl"

# Define model and data parameters
SEED = 42
TEST_SIZE = 0.2
TARGET = "fatigue" # This must match the cleaned column name