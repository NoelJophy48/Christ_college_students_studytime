import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "Electric_Bill_fan.pkl"
poly,model = joblib.load(model_path)

st.title("Electricity bill predictor")
st.write("Enter the AC and Fan units to predict the bill.")


ac_units = st.number_input("AC_Units", min_value=0.0, step=0.5)
fan_units = st.number_input("Fan_Units", min_value=0.0, step=0.5)

if st.button("Predict"):
	input_data = pd.DataFrame({"AC_Units": [ac_units],"Fan_Units": [fan_units]})
	input_data_poly = poly.transform(input_data)
	prediction = model.predict(input_data_poly)[0]

	st.success(f"Predicted Bill: {prediction:.0f} ")
