import joblib

# Load model
model = joblib.load("model/aqi_model.pkl")

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

def suggest_actions(category):
    suggestions = {
        "Good": "Air quality is safe.",
        "Satisfactory": "Reduce vehicle usage.",
        "Moderate": "Use public transport.",
        "Poor": "Wear masks and avoid outdoor activity.",
        "Very Poor": "Work from home if possible.",
        "Severe": "Emergency measures needed."
    }
    return suggestions[category]

# Example input
input_data = [[120, 180, 40, 10, 1.2, 30, 210, 200]]

prediction = model.predict(input_data)[0]
category = aqi_category(prediction)
suggestion = suggest_actions(category)

print("Predicted AQI:", prediction)
print("Category:", category)
print("Suggestion:", suggestion)
