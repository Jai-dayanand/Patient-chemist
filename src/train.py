# src/train.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from src.config import MODEL_PATH, SEED, PROCESSED_PATH

def train_models():
    """
    Trains multiple regression models, performs hyperparameter tuning,
    and saves the best model.
    """
    X_train = pd.read_csv(os.path.join(PROCESSED_PATH, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_PATH, 'y_train.csv')).values.ravel()

    models = {
        "RandomForest": RandomForestRegressor(random_state=SEED),
        "XGBoost": xgb.XGBRegressor(random_state=SEED),
        "LightGBM": lgb.LGBMRegressor(random_state=SEED),
    }

    param_grids = {
        "RandomForest": {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None]
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        "LightGBM": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50]
        }
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        print(f"Training {name}...")
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        grid_search = GridSearchCV(model, param_grids[name], cv=kf, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Best score for {name}: {grid_search.best_score_}")
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_

    print(f"\nBest model: {type(best_model).__name__} with R² score: {best_score}")

    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train_models()