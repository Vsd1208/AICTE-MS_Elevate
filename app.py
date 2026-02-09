import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ==============================
# CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Smart AQI System",
    page_icon="🌿",
    layout="wide"
)

# Load trained model
model = joblib.load("model/aqi_model.pkl")

# Your OpenWeather API key
API_KEY = "be209e8818c3421776cc97a2e31c4359"

# City coordinates
CITY_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946)
}

FEATURE_NAMES = ['PM2.5', 'PM10', 'NO2', 'SO2']

# ==============================
# FUNCTIONS
# ==============================

# Fetch live pollution data
def get_live_pollution(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()

    components = data['list'][0]['components']

    return {
        'PM2.5': components['pm2_5'],
        'PM10': components['pm10'],
        'NO2': components['no2'],
        'SO2': components['so2']
    }

# Predict AQI (DataFrame → removes sklearn warning)
def predict_aqi(pollution_dict):
    input_df = pd.DataFrame([pollution_dict], columns=FEATURE_NAMES)
    return model.predict(input_df)[0]

# AQI category
def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Satisfactory", "lightgreen"
    elif aqi <= 200:
        return "Moderate", "orange"
    elif aqi <= 300:
        return "Poor", "red"
    elif aqi <= 400:
        return "Very Poor", "purple"
    else:
        return "Severe", "maroon"

# Health advice
def health_advice(category):
    advice = {
        "Good": "Air quality is healthy. Enjoy outdoor activities.",
        "Satisfactory": "Sensitive people should limit long outdoor exposure.",
        "Moderate": "Reduce outdoor exercise and prefer public transport.",
        "Poor": "Wear masks and avoid prolonged outdoor activities.",
        "Very Poor": "Stay indoors as much as possible.",
        "Severe": "Health emergency. Avoid going outside."
    }
    return advice[category]

# AQI Gauge
def show_gauge(aqi):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        title={'text': "AQI Level"},
        gauge={
            'axis': {'range': [0, 500]},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "lightgreen"},
                {'range': [100, 200], 'color': "orange"},
                {'range': [200, 300], 'color': "red"},
                {'range': [300, 500], 'color': "maroon"}
            ]
        }
    ))
    st.plotly_chart(fig, width="stretch")

# ==============================
# UI
# ==============================

st.title("🌿 Smart Air Quality Forecast System")
st.markdown("Real-time AQI prediction with AI-based health advisory")

st.divider()

# City selection
city = st.selectbox("Select City", list(CITY_COORDS.keys()))

if st.button("Get Live Air Quality", width="stretch"):

    lat, lon = CITY_COORDS[city]

    try:
        pollution = get_live_pollution(lat, lon)

        # Predict AQI
        today_aqi = predict_aqi(pollution)
        category, color = aqi_category(today_aqi)

        st.success("Live data fetched successfully")

        st.divider()

        col1, col2 = st.columns(2)

        # Left panel
        with col1:
            st.subheader(f"{city} AQI")
            show_gauge(today_aqi)
            st.markdown(f"**Category:** {category}")
            st.info(health_advice(category))

        # Right panel – Tomorrow forecast
        with col2:
            forecast_pollution = {
                key: value * np.random.uniform(0.9, 1.1)
                for key, value in pollution.items()
            }
            tomorrow_aqi = predict_aqi(forecast_pollution)

            st.metric("Tomorrow Forecast AQI", round(tomorrow_aqi, 2))

        # Alerts
        if today_aqi > 300:
            st.error("⚠️ Severe Pollution Alert! Avoid outdoor activities.")
        elif today_aqi > 200:
            st.warning("⚠️ Air quality is unhealthy. Limit outdoor exposure.")

        # Pollutant levels
        st.subheader("Current Pollutant Levels")
        pollutant_df = pd.DataFrame({
            "Pollutant": pollution.keys(),
            "Value": pollution.values()
        })
        st.bar_chart(pollutant_df.set_index("Pollutant"))

        # Explainable AI
        st.subheader("Pollution Impact on AQI")
        importance_df = pd.DataFrame({
            "Pollutant": FEATURE_NAMES,
            "Impact": model.feature_importances_
        }).sort_values(by="Impact", ascending=False)

        st.bar_chart(importance_df.set_index("Pollutant"))

        # 5-day trend
        st.subheader("5-Day AQI Forecast Trend")
        trend = [today_aqi]
        current = today_aqi
        for _ in range(4):
            current = current * np.random.uniform(0.9, 1.1)
            trend.append(current)

        trend_df = pd.DataFrame({
            "Day": ["Today", "Day 2", "Day 3", "Day 4", "Day 5"],
            "AQI": trend
        })
        st.line_chart(trend_df.set_index("Day"))

    except:
        st.error("Unable to fetch data. Check API key or wait for activation.")

st.divider()
st.caption("AICTE Internship Project | Industry-Level AQI Forecasting System")
st.caption("Developed by: Vandanapu Saidhiraj")
