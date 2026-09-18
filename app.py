import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "Electric_Bill.pkl"
model = joblib.load(model_path)

st.title("AC Units")
st.write("Enter the AC units to predict the bill.")

ac_units = st.number_input("AC_Units", min_value=0.0, step=0.5)

if st.button("Predict"):
	input_data = pd.DataFrame({"AC_Units": [ac_units]})
	prediction = model.predict(input_data)[0]

	if prediction == 1:
		st.success(f"Predicted Bill: {prediction:.0f} ")
	else:
		st.error("Error Occurred")
