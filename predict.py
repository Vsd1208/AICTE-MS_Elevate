import joblib

# Load trained model
model = joblib.load("model/aqi_model.pkl")

# AQI Category Function
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

# Suggestion Function
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

# Example Input
# Order: PM2.5, PM10, NO2, SO2
input_data = [[120, 180, 40, 10]]

# Predict
predicted_aqi = model.predict(input_data)[0]
category = aqi_category(predicted_aqi)
suggestion = suggest_actions(category)

print("Predicted AQI:", round(predicted_aqi, 2))
print("Category:", category)
print("Suggestion:", suggestion)
