import streamlit as st
import joblib

model = joblib.load("model/aqi_model.pkl")

def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

st.title("AQI Prediction System")

pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")
o3 = st.number_input("O3")
lag1 = st.number_input("Yesterday AQI")
lag2 = st.number_input("Day Before AQI")

if st.button("Predict"):
    data = [[pm25, pm10, no2, so2, co, o3, lag1, lag2]]
    aqi = model.predict(data)[0]
    category = aqi_category(aqi)
    
    st.write("Predicted AQI:", aqi)
    st.write("Category:", category)
