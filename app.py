# ============================================================
# STEP 1: Import Libraries
# ============================================================
import numpy as np
import streamlit as st
import pickle

# ============================================================
# STEP 2: Load Saved Model and Scaler
# ============================================================
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ============================================================
# STEP 3: App Title and Description
# ============================================================
st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a patient is **Diabetic or Not** based on medical input features.")
st.write("---")

# ============================================================
# STEP 4: Sidebar — Patient Input Form
# ============================================================
st.sidebar.header("Enter Patient Data")

pregnancies            = st.sidebar.number_input("Pregnancies",            min_value=0,   max_value=20,   value=1,    step=1)
glucose                = st.sidebar.number_input("Glucose",                min_value=0,   max_value=200,  value=120,  step=1)
blood_pressure         = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0,   max_value=150,  value=70,   step=1)
skin_thickness         = st.sidebar.number_input("Skin Thickness (mm)",    min_value=0,   max_value=100,  value=20,   step=1)
insulin                = st.sidebar.number_input("Insulin (mu U/ml)",      min_value=0,   max_value=900,  value=80,   step=1)
bmi                    = st.sidebar.number_input("BMI",                    min_value=0.0, max_value=70.0, value=25.0, step=0.1)
diabetes_pedigree      = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age                    = st.sidebar.number_input("Age",                    min_value=1,   max_value=120,  value=30,   step=1)

# ============================================================
# STEP 5: Display Entered Values in Main Panel
# ============================================================
st.subheader("📋 Patient Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pregnancies",  pregnancies)
col2.metric("Glucose",      glucose)
col3.metric("Blood Pressure", blood_pressure)
col4.metric("Skin Thickness", skin_thickness)

col5, col6, col7, col8 = st.columns(4)
col5.metric("Insulin",      insulin)
col6.metric("BMI",          bmi)
col7.metric("Pedigree Fn.", diabetes_pedigree)
col8.metric("Age",          age)

st.write("---")

# ============================================================
# STEP 6: Preprocess Input (Scale it)
# ============================================================
input_data = np.array([
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, diabetes_pedigree, age
]).reshape(1, -1)

scaled_input = scaler.transform(input_data)

# ============================================================
# STEP 7: Predict on Button Click
# ============================================================
if st.button("🔍 Predict Result"):
    prediction    = model.predict(scaled_input)
    probability   = model.predict_proba(scaled_input)

    diabetic_prob    = round(probability[0][1] * 100, 2)
    no_diabetic_prob = round(probability[0][0] * 100, 2)

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ **Diabetic** — The model predicts this patient has diabetes.")
    else:
        st.success(f"✅ **Not Diabetic** — The model predicts this patient does NOT have diabetes.")

    # Show confidence probabilities
    st.write("#### Confidence Scores:")
    st.progress(int(diabetic_prob))
    col_a, col_b = st.columns(2)
    col_a.metric("🔴 Diabetic Probability",    f"{diabetic_prob}%")
    col_b.metric("🟢 No Diabetes Probability", f"{no_diabetic_prob}%")

# ============================================================
# STEP 8: Footer
# ============================================================
st.write("---")
st.caption("⚕️ This app is for educational purposes only. Always consult a medical professional.")
