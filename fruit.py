import streamlit as st
import pandas as pd
import pickle
import tensorflow

st.sidebar.title("Choose features")

mass = st.sidebar.number_input("Enter mass:", 0)
width = st.sidebar.number_input("Enter width:", step=0.1, value=0.0, min_value=0.00, max_value=None)
height = st.sidebar.number_input("Enter height:",  step=0.1, value=0.0, min_value=0.00, max_value=None)
clrscr = st.sidebar.slider("Enter color score:", 0.00, 1.00)

st.header("Model Deployment")

my_dict= {
    "mass": mass,
    "width": width,
    "height": height,
    "color_score": clrscr
}

inputs =pd.DataFrame.from_dict([my_dict])

from joblib import load
encoder = load('encoder.joblib')
model = load('final_fruit_model.joblib')

st.table(inputs)

if st.button("Predict"):
    pred = encoder.inverse_transform(model.predict(inputs))
    st.success(pred)