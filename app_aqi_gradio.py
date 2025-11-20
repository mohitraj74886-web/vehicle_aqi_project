import gradio as gr
import cv2
import tempfile
import pandas as pd
import numpy as np
import requests
import shutil
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load YOLO model once
model = YOLO("yolov8n.pt")

# -------------------------------------------
# AQI Classification + Color Box
# -------------------------------------------
def classify_aqi(aqi):
    if aqi <= 50:
        return "Good 🙂", "#4CAF50"      # Green
    elif aqi <= 100:
        return "Moderate 😐", "#FFEB3B"  # Yellow
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups 😷", "#FF9800"  # Orange
    elif aqi <= 200:
        return "Unhealthy 😨", "#F44336"  # Red
    elif aqi <= 300:
        return "Very Unhealthy ☠️", "#9C27B0"  # Purple
    else:
        return "Hazardous 🚫", "#6D4C41"  # Brown


# -------------------------------------------
# Fetch REAL AQI (PM2.5 → AQI approx.)
# -------------------------------------------
def fetch_real_aqi(api_key, lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        data = requests.get(url).json()

        pm25 = data["list"][0]["components"]["pm2_5"]
        pm10 = data["list"][0]["components"]["pm10"]

        # Convert PM2.5 to approximate US AQI
        aqi = int(pm25 * 4)

        return aqi, pm25, pm10
    except:
        return None, None, None


# -------------------------------------------
# Fetch REAL weather
# -------------------------------------------
def fetch_real_weather(api_key, lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
        data = requests.get(url).json()

        temp = data["main"]["temp"] - 273.15
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]

        return temp, humidity, wind
    except:
        return None, None, None


# -------------------------------------------
# MAIN PROCESS FUNCTION
# -------------------------------------------
def process_video(video_file, api_key, lat, lon):

    if video_file is None:
        return "No video uploaded.", None, None

    # ----- FIXED: Copy video to temp path (no .read()) -----
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    shutil.copy(video_file.name, temp_path)
    # --------------------------------------------------------

    # Read video
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    detections = []
    frame_num = 0
    sample_stride = 3

    # YOLO detection per frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_stride == 0:
            results = model(frame)
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if conf > 0.4 and label in ("car", "truck", "bus", "motorbike"):
                    detections.append(frame_num)

        frame_num += 1

    cap.release()

    if len(detections) == 0:
        return "No vehicles detected.", None, None

    # Vehicle counts per second
    df = pd.DataFrame({"frame": detections})
    df["time_sec"] = (df["frame"] / fps).astype(int)
    counts = df.groupby("time_sec").size()

    # Fetch REAL AQI + weather
    real_aqi, pm25, pm10 = fetch_real_aqi(api_key, lat, lon)
    temp, humidity, wind = fetch_real_weather(api_key, lat, lon)

    if real_aqi is None:
        return "API error: Check your API key or internet.", None, None

    # AQI category + color
    aqi_label, aqi_color = classify_aqi(real_aqi)

    # Feature DataFrame
    df_feat = pd.DataFrame({
        "time_sec": counts.index,
        "vehicle_count": counts.values,
        "temp": temp,
        "humidity": humidity,
        "wind": wind,
        "AQI": real_aqi
    })

    # Train ML model
    X = df_feat[["vehicle_count", "temp", "humidity", "wind"]]
    y = df_feat["AQI"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = LinearRegression()
    reg.fit(X_scaled, y)
    predicted = reg.predict(X_scaled)

    # Plot
    plt.figure(figsize=(7, 3))
    plt.plot(df_feat["time_sec"], df_feat["AQI"], label="Real AQI", linewidth=2)
    plt.plot(df_feat["time_sec"], predicted, "--", label="Predicted AQI")
    plt.xlabel("Time (sec)")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()

    # ---- COLOR BOX HTML ----
    color_box_html = f"""
    <div style='padding: 18px; border-radius: 10px; background-color: {aqi_color};
                color: black; font-size: 22px; font-weight: bold; text-align: center;'>
        AQI: {real_aqi} — {aqi_label}
    </div>
    """

    return (
        f"PM2.5: {pm25}\nPM10: {pm10}\nTemperature: {temp:.2f}°C\nHumidity: {humidity}%",
        plt,
        color_box_html
    )


# -------------------------------------------
# GRADIO INTERFACE (WITH COLOR BOX)
# -------------------------------------------
interface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.File(label="Upload Traffic Video (.mp4)"),
        gr.Textbox(label="OpenWeatherMap API Key"),
        gr.Number(label="Latitude", value=28.6139),
        gr.Number(label="Longitude", value=77.2090)
    ],
    outputs=[
        gr.Textbox(label="Weather & PM Info"),
        gr.Plot(label="AQI Prediction Plot"),
        gr.HTML(label="AQI Category")
    ],
    title="Real AQI Predictor from Traffic Video",
    description="Counts vehicles using YOLO, fetches real AQI & weather, predicts AQI, and shows AQI category with colored box."
)

interface.launch()
