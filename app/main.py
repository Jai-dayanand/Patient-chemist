# app/main.py

import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
from src.config import MODEL_PATH, PROCESSED_PATH, SHAP_PATH

st.title("The Precision Chemist: Steel Fatigue Strength Prediction")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(os.path.join(PROCESSED_PATH, 'scaler.pkl'))
X_train_columns = pd.read_csv(os.path.join(PROCESSED_PATH, 'X_train.csv')).columns

st.sidebar.header("Input Chemical Composition")
input_features = {}
for feature in X_train_columns:
    input_features[feature] = st.sidebar.number_input(feature, value=0.0, format="%.4f")

if st.sidebar.button("Predict Fatigue Strength"):
    input_df = pd.DataFrame([input_features])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    st.subheader("Predicted Fatigue Strength")
    st.write(f"{prediction[0]:.2f} MPa")

    st.subheader("Prediction Explanation (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                          base_values=explainer.expected_value,
                                          data=scaled_input[0],
                                          feature_names=X_train_columns.tolist()),
                        show=False)
    st.pyplot(fig)