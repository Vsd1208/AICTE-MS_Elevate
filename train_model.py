import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os


file_path = "data/air_quality.csv"

data = pd.read_csv(
    file_path,
    encoding="latin1",
    low_memory=False
)


# Step 2: Clean Column Names

# Remove spaces and strange BOM characters
data.columns = data.columns.str.strip().str.replace('ï»¿', '', regex=False)

print("Available Columns:")
print(data.columns)

# Step 3: Select Required Columns

required_columns = ['pm2_5', 'rspm', 'no2', 'so2']

# Check if columns exist
for col in required_columns:
    if col not in data.columns:
        print(f"Column missing: {col}")

data = data[required_columns]

# Rename for clarity
data.rename(columns={
    'pm2_5': 'PM2.5',
    'rspm': 'PM10',
    'no2': 'NO2',
    'so2': 'SO2'
}, inplace=True)

data.dropna(inplace=True)

print("Data after cleaning:", data.shape)

data['AQI'] = data['PM2.5']

features = ['PM2.5', 'PM10', 'NO2', 'SO2']
target = 'AQI'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/aqi_model.pkl")

print("\nModel trained and saved at: model/aqi_model.pkl")