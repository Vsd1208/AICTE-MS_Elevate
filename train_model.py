import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = pd.read_csv("data/air_quality.csv")

# Select useful columns
data = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']]

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Create lag features
data['AQI_lag1'] = data['AQI'].shift(1)
data['AQI_lag2'] = data['AQI'].shift(2)
data.dropna(inplace=True)

# Features and target
X = data[['PM2.5','PM10','NO2','SO2','CO','O3','AQI_lag1','AQI_lag2']]
y = data['AQI']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/aqi_model.pkl")

print("Model trained and saved!")
