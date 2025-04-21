import streamlit as st
import joblib
import pandas as pd

# Load model and feature list
model = joblib.load('readmission_model.pkl')
features = joblib.load('feature_list.pkl')

# Set page config
st.set_page_config(page_title="Patient Readmission Predictor", layout="centered")
st.title("🏥 Patient Readmission Prediction App")
st.markdown("Enter the patient's details below to predict if they will be readmitted.")

# Define which features are binary (0/1)
binary_features = ['insulin', 'metformin', 'discharge_disposition_id_2', 'admission_type_id_3']

# Input form
user_input = {}
for feature in features:
    if feature in binary_features:
        user_input[feature] = st.selectbox(
            f"{feature} (Yes = 1, No = 0)",
            [0, 1],
            format_func=lambda x: 'Yes' if x == 1 else 'No'
        )
    else:
        user_input[feature] = st.number_input(f"{feature}", step=1.0)

# Predict button
if st.button("Predict Readmission"):
    input_df = pd.DataFrame([user_input])

    # Predict
    pred_proba = model.predict_proba(input_df)[0][1]
    threshold = 0.3
    prediction = 1 if pred_proba >= threshold else 0

    if prediction == 1:
        st.error(f"🔴 Patient is likely to be readmitted (Probability: {pred_proba:.2f})")
        st.markdown(
            """
            **Why?**
            - High medication count or long hospital stay might indicate a more serious condition.
            - Changes in insulin or treatment plans could increase readmission risk.
            - More frequent encounters or diagnoses often signal ongoing complications.
            """
        )
    else:
        st.success(f"🟢 Patient is NOT likely to be readmitted (Probability: {pred_proba:.2f})")
        st.markdown(
            """
            **Why?**
            - Fewer medications and shorter hospital stay suggest a stable condition.
            - No recent changes in insulin or treatment plan may reflect steady recovery.
            - Lower number of diagnoses or encounters indicates better patient health.
            """
        )
