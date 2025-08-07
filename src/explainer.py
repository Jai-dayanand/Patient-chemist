# src/explainer.py

import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
from src.config import MODEL_PATH, SHAP_PATH, PROCESSED_PATH

def explain_model():
    """
    Uses SHAP to explain the model's predictions and saves the plots.
    """
    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(os.path.join(PROCESSED_PATH, 'X_train.csv'))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if not os.path.exists(os.path.dirname(SHAP_PATH)):
        os.makedirs(os.path.dirname(SHAP_PATH))
    joblib.dump(shap_values, SHAP_PATH)
    print(f"SHAP values saved to {SHAP_PATH}")

    # Global feature importance plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.title('SHAP Global Feature Importance')
    plt.savefig('shap_summary_plot.png')
    plt.show()

    # Waterfall plot for a single prediction
    plt.figure()
    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                          base_values=explainer.expected_value,
                                          data=X_train.iloc[0],
                                          feature_names=X_train.columns.tolist()),
                        show=False)
    plt.tight_layout()
    plt.savefig('shap_waterfall_plot.png')
    plt.show()

if __name__ == '__main__':
    explain_model()