# app/main.py

import sys
import os
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

# --- Robust Path Fix ---
# This block of code ensures that the 'src' module can be found by Python,
# no matter where you run the 'streamlit' command from.
try:
    # This works when running from the project root (streamlit run app/main.py)
    from src.config import MODEL_PATH, PROCESSED_PATH
except ModuleNotFoundError:
    # This is a fallback for other execution scenarios
    # Get the absolute path of the current script (main.py)
    current_script_path = os.path.abspath(__file__)
    # Get the directory containing the script (the 'app' folder)
    app_dir = os.path.dirname(current_script_path)
    # Get the parent directory of 'app' (this is the project root)
    project_root = os.path.dirname(app_dir)
    # Add the project root to Python's system path
    sys.path.insert(0, project_root)
    # Now, the import should work
    from src.config import MODEL_PATH, PROCESSED_PATH

# --- Streamlit App ---

st.set_page_config(page_title="The Precision Chemist", layout="wide")
st.title("🔬 The Precision Chemist")
st.subheader("Predicting Steel Fatigue Strength from Chemical Composition")

# --- Function to load artifacts ---
# Using st.cache_data ensures these heavy objects are loaded only once.
@st.cache_data
def load_artifacts():
    """Loads the machine learning model, scaler, and training columns."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(os.path.join(PROCESSED_PATH, 'scaler.pkl'))
    # Load column names from the processed training data file
    try:
        train_cols = pd.read_csv(os.path.join(PROCESSED_PATH, 'X_train.csv')).columns
    except FileNotFoundError:
        st.error(
            "Processed data not found. Please run the training pipeline first "
            "by executing the scripts in the 'notebooks' or 'src' folder."
        )
        return None, None, None
    return model, scaler, train_cols

# --- Main Application Logic ---
model, scaler, X_train_columns = load_artifacts()

if model is None:
    st.stop()

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Chemical Composition (%)")

input_features = {}
# Create number inputs for each feature in the sidebar
for feature in X_train_columns:
    input_features[feature] = st.sidebar.number_input(
        label=feature,
        value=0.0,      # Default value
        min_value=0.0,
        max_value=100.0, # Reasonable max for percentage
        format="%.4f"   # Format for up to 4 decimal places
    )

# --- Prediction and Explanation ---
if st.sidebar.button("Predict Fatigue Strength", type="primary"):
    # Create a DataFrame from the user's inputs
    input_df = pd.DataFrame([input_features])
    
    # Ensure the columns are in the same order as when the model was trained
    input_df = input_df[X_train_columns]

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(scaled_input)

    # --- Display Results ---
    st.header("Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Fatigue Strength",
            value=f"{prediction[0]:.2f} MPa"
        )
        st.write("This value represents the model's prediction for the steel's resistance to failure under repeated loading.")

    with col2:
        st.info("What is Fatigue Strength?", icon="❓")
        st.write(
            "Fatigue strength is the highest stress that a material can withstand for a given number of cycles without breaking. "
            "It is a critical property for materials used in components subjected to vibrations or fluctuating loads, like engine parts, bridges, and aircraft structures."
        )
    
    st.divider()

    # --- SHAP Explanation ---
    st.header("Prediction Explanation (SHAP)")
    st.write(
        "The chart below shows how each chemical element contributed to the final prediction. "
        "Features in **red** pushed the prediction higher, while those in **blue** pushed it lower."
    )

    # Use a TreeExplainer for tree-based models (like RandomForest, XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    # Create a new figure for the SHAP plot
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=scaled_input[0],
            feature_names=X_train_columns.tolist()
        ),
        show=False # We will render it with st.pyplot
    )
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Enter chemical values in the sidebar and click 'Predict Fatigue Strength' to see the results.")