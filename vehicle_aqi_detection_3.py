"""
DAY 3 — AQI Prediction Model (Correct Free API Endpoints)
----------------------------------------------------------
Uses:
✔ Vehicle counts (from Day 2)
✔ Real AQI (free endpoint)
✔ Real Temperature, Humidity, Wind (free endpoint)
✔ Safe fallback when API fails
✔ No NaNs
✔ Works for short 10-second video

Endpoints used:
AQI (free):
    http://api.openweathermap.org/data/2.5/air_pollution

Weather (free):
    http://api.openweathermap.org/data/2.5/weather
"""

import os
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
VEH_COUNTS_CSV = "vehicle_counts_timeseries.csv"
OUTPUT_MODEL = "trained_model.joblib"
PLOT_FILE = "aqi_prediction_plot.png"

USE_REAL_DATA = True       # Set True to use the APIs
OWM_API_KEY = "0ef40d1d4a108bea8453bc3638ed1af9"       # <-- PUT YOUR KEY HERE

# Your city coordinates
LAT = 15.870032  # New Delhi
LON = 100.992541

# --------------------------------------------------------------
# 1) LOAD VEHICLE COUNTS
# --------------------------------------------------------------
if not os.path.exists(VEH_COUNTS_CSV):
    raise FileNotFoundError("Run Day 2 first to generate vehicle_counts_timeseries.csv")

df_counts = pd.read_csv(VEH_COUNTS_CSV)
df_counts["time_sec"] = df_counts["time_sec"].astype(float)

start_time = pd.Timestamp.now().floor("s")
df_counts["datetime"] = df_counts["time_sec"].apply(lambda s: start_time + pd.Timedelta(seconds=s))
df_counts = df_counts.set_index("datetime").sort_index()

# 1-second bins
counts_per_sec = df_counts["vehicle_count"].resample("1s").sum().fillna(0)
counts_per_sec = counts_per_sec.rename("vehicle_count")

print("\nCounts per second:")
print(counts_per_sec.head())

# --------------------------------------------------------------
# 2) REAL AQI FETCH (Free endpoint)
# --------------------------------------------------------------
def fetch_real_aqi(index):
    """Fetch current AQI using free endpoint (one call per second)."""
    aqi_values = []
    for _ in index:
        try:
            url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OWM_API_KEY}"
            r = requests.get(url, timeout=10)
            data = r.json()
            # AQI returned as index 1–5
            aqi_idx = data["list"][0]["main"]["aqi"]
            mapping = {1: 30, 2: 60, 3: 100, 4: 180, 5: 250}
            aqi_values.append(mapping.get(aqi_idx, 100))
        except:
            aqi_values.append(np.nan)
    return pd.Series(aqi_values, index=index)

def simulate_aqi(vc):
    rng = np.random.default_rng(42)
    base = 50
    noise = rng.normal(0, 5, size=len(vc))
    simulated = base + 0.08 * vc.values + noise
    return pd.Series(np.clip(simulated, 10, 300), index=vc.index, name="AQI")

if USE_REAL_DATA and OWM_API_KEY != "YOUR_API_KEY_HERE":
    print("\nFetching REAL AQI...")
    aqi_series = fetch_real_aqi(counts_per_sec.index)
    if aqi_series.isna().mean() > 0.4:
        print("Too many missing values → Using simulated AQI.")
        aqi_series = simulate_aqi(counts_per_sec)
else:
    print("\nUsing SIMULATED AQI...")
    aqi_series = simulate_aqi(counts_per_sec)

# --------------------------------------------------------------
# 3) REAL WEATHER FETCH (Free endpoint)
# --------------------------------------------------------------
def fetch_real_weather(index):
    temps = []
    hums = []
    winds = []

    for _ in index:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OWM_API_KEY}"
            r = requests.get(url, timeout=10)
            data = r.json()

            # Temperature returned in Kelvin → convert to Celsius
            temp = data["main"]["temp"] - 273.15
            humidity = data["main"]["humidity"]
            wind = data["wind"]["speed"]

            temps.append(temp)
            hums.append(humidity)
            winds.append(wind)

        except:
            temps.append(np.nan)
            hums.append(np.nan)
            winds.append(np.nan)

    return pd.DataFrame({
        "temp": temps,
        "humidity": hums,
        "wind": winds
    }, index=index)

def simulate_weather(index):
    rng = np.random.default_rng(123)
    hrs = np.array([t.hour for t in index])
    temp = 25 + 3*np.sin((hrs/24)*2*np.pi) + rng.normal(0, 1, len(index))
    hum = 60 + 10*np.cos((hrs/24)*2*np.pi) + rng.normal(0, 2, len(index))
    wind = 2 + np.sin((hrs/24)*2*np.pi) + rng.normal(0, 0.3, len(index))

    return pd.DataFrame({"temp": temp, "humidity": hum, "wind": wind}, index=index)

if USE_REAL_DATA and OWM_API_KEY != "YOUR_API_KEY_HERE":
    print("\nFetching REAL weather...")
    weather_df = fetch_real_weather(counts_per_sec.index)

    if weather_df.isna().mean().mean() > 0.3:
        print("Weather API failed → Using simulated weather.")
        weather_df = simulate_weather(counts_per_sec.index)
else:
    print("\nUsing SIMULATED weather...")
    weather_df = simulate_weather(counts_per_sec.index)

# --------------------------------------------------------------
# 4) BUILD FINAL DATAFRAME
# --------------------------------------------------------------
df = pd.DataFrame(index=counts_per_sec.index)
df["vehicle_count"] = counts_per_sec
df["vc_roll_3"] = df["vehicle_count"].rolling(3, min_periods=1).mean()
df["vc_roll_5"] = df["vehicle_count"].rolling(5, min_periods=1).mean()
df["hour"] = df.index.hour

df = df.join(weather_df)
df["AQI"] = aqi_series

# Fill NaN values safely
df = df.ffill().bfill()

print("\nFinal Combined Table:")
print(df.head())

# --------------------------------------------------------------
# 5) MODEL TRAINING
# --------------------------------------------------------------
FEATURES = ["vehicle_count", "vc_roll_3", "vc_roll_5",
            "hour", "temp", "humidity", "wind"]

X = df[FEATURES].values
y = df["AQI"].values

train_size = int(len(df) * 0.7)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale only for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Evaluation helper
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("\nMODEL PERFORMANCE:")
print("Linear Regression:", "MAE=", mean_absolute_error(y_test, lr_pred), 
      "RMSE=", rmse(y_test, lr_pred),
      "R2=", r2_score(y_test, lr_pred))

print("Random Forest:", "MAE=", mean_absolute_error(y_test, rf_pred), 
      "RMSE=", rmse(y_test, rf_pred),
      "R2=", r2_score(y_test, rf_pred))

# Pick best model
if rmse(y_test, rf_pred) < rmse(y_test, lr_pred):
    joblib.dump({"model": rf, "features": FEATURES}, OUTPUT_MODEL)
else:
    joblib.dump({"model": lr, "scaler": scaler, "features": FEATURES}, OUTPUT_MODEL)

print(f"\nModel saved to {OUTPUT_MODEL}")

# --------------------------------------------------------------
# 6) PLOT PREDICTION
# --------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(df.index[train_size:], y_test, label="Actual AQI", linewidth=2)
plt.plot(df.index[train_size:], lr_pred, "--", label="LR Pred")
plt.plot(df.index[train_size:], rf_pred, ":", label="RF Pred")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("AQI Prediction (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.show()

print(f"Plot saved as {PLOT_FILE}")
