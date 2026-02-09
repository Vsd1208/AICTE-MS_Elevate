import streamlit as st
import joblib

# Load model
model = joblib.load("model/aqi_model.pkl")

# AQI Category
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# Suggestions
def suggest_actions(category):
    suggestions = {
        "Good": "Air quality is safe. Maintain green practices.",
        "Satisfactory": "Reduce vehicle usage.",
        "Moderate": "Use public transport and avoid outdoor burning.",
        "Poor": "Wear mask and avoid outdoor activities.",
        "Very Poor": "Work from home if possible.",
        "Severe": "Emergency measures needed. Avoid going outside."
    }
    return suggestions[category]

st.title("AI-based AQI Prediction System")

pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")

if st.button("Predict AQI"):
    data = [[pm25, pm10, no2, so2]]
    aqi = model.predict(data)[0]
    category = aqi_category(aqi)
    suggestion = suggest_actions(category)

    st.subheader(f"Predicted AQI: {round(aqi,2)}")
    st.subheader(f"Category: {category}")
    st.write("Recommendation:", suggestion)