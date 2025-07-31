import streamlit as st
import numpy as np
import joblib

# Model selection
st.title("SFRC Strength Prediction App")
st.markdown("Select the target property to predict:")

# Choose model type
model_option = st.selectbox("Prediction Target", ["Compressive Strength (CS)", "Splitting Tensile Strength (ST)", "Flexural Strength (FC)"])

# Map model to file
model_map = {
    "Compressive Strength (CS)": r"C:\Users\SEJONG\OneDrive\01. Concrete\06. Data analysis\Python\Expand\CatBoost_optimized_CS.pkl",
    "Splitting Tensile Strength (ST)": r"C:\Users\SEJONG\OneDrive\01. Concrete\06. Data analysis\Python\Expand\CatBoost_optimized_ST.pkl",
    "Flexural Strength (FC)": r"C:\Users\SEJONG\OneDrive\01. Concrete\06. Data analysis\Python\Expand\CatBoost_optimized_FC.pkl"
}

# Load selected model
model = joblib.load(model_map[model_option])

# Feature input
st.markdown("Enter mix design and fiber parameters:")

W = st.number_input("W - Water (kg)", value=200.0)
C = st.number_input("C - Cement (kg)", value=400.0)
S = st.number_input("S - Sand (kg)", value=600.0)
CA = st.number_input("CA - Coarse Aggregate (kg)", value=1000.0)
s = st.number_input("s - Maximum Aggregate Size (mm)", value=20.0)
SP = st.number_input("SP - Admixture (kg)", value=5.0)
pf = st.selectbox("pf - Fiber Shape", options=[0, 1, 2], format_func=lambda x: {0: "Hooked", 1: "Straight", 2: "Corrugated"}.get(x))
Vf = st.number_input("Vf - Fiber Volume Content (%)", value=1.0)
df = st.number_input("df - Fiber Diameter (mm)", value=0.75)
Lf = st.number_input("Lf - Fiber Length (mm)", value=30.0)

input_data = np.array([[W, C, S, CA, s, SP, pf, Vf, df, Lf]])

# Prediction
if st.button("Predict"):
    result = model.predict(input_data)[0]
    st.success(f"Predicted {model_option}: **{result:.2f} MPa**")
