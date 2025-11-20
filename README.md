This project predicts Air Quality Index (AQI) using traffic video analysis, real-time weather, and real PM2.5/PM10 measurements.
It uses YOLOv8 to detect vehicles in a traffic video, counts the number of cars/trucks/buses/bikes over time, fetches live AQI and weather from OpenWeatherMap API, and trains a lightweight ML model to estimate AQI trends.

A clean Gradio-based UI allows users to upload a video, enter location coordinates, and view predictions along with a color-coded AQI category box.

✨ Features
🧠 1. Computer Vision

Uses YOLOv8 to detect vehicles in real-time.

Counts cars, trucks, buses, and motorbikes.

Calculates vehicle density per second.

🌍 2. Real-Time API Integration

Fetches PM2.5 and PM10 using OpenWeather Air Pollution API.

Fetches temperature, humidity, and wind speed using OpenWeather Weather API.

📈 3. Machine Learning Model

Trains a Linear Regression model to predict AQI trend from:

Vehicle count

Temperature

Humidity

Wind speed

Generates AQI prediction plot.

🎨 4. AQI Classification (US EPA Standard)

Shows AQI category using a clear color-coded box:

🟩 Good

🟨 Moderate

🟧 Unhealthy for Sensitive Groups

🟥 Unhealthy

🟪 Very Unhealthy

🟫 Hazardous

💡 5. User-Friendly Gradio Web App

Upload video file (.mp4)

Enter API key, latitude, longitude

Get real AQI, PM levels, weather, and ML-based AQI prediction.
📦 Installation
1. Clone this repository
git clone https://github.com/YOUR_USERNAME/vehicle_aqi_project.git
cd vehicle_aqi_project

2. Install dependencies
pip install -r requirements.txt

3. Download YOLOv8 model (if not already included)
yolo download yolo8n.pt

▶️ Usage
Run the Gradio app:
python aqi_app_real_color.py


This will open a local web interface at:

http://127.0.0.1:7860

In the UI:

Upload a traffic video (10–20 sec recommended)

Enter your OpenWeatherMap API Key

Enter latitude & longitude

Click Submit

🧪 How It Works
1️⃣ Vehicle Detection

YOLOv8 processes frames and counts:

Car

Truck

Bus

Motorbike

2️⃣ Get Real AQI & Weather

OpenWeather API provides:

PM2.5

PM10

Temperature

Humidity

Wind speed

3️⃣ Train AQI Prediction Model

Features used:

traffic density

weather conditions

pollutant levels

Model output:

Real AQI

Predicted AQI trend

4️⃣ Visualization

The app displays:

AQI prediction graph

Weather information

PM2.5 / PM10 values

AQI category with color box

🌈 AQI Categories (US EPA Standard)
AQI Range	Category	Color
0–50	Good	🟩 Green
51–100	Moderate	🟨 Yellow
101–150	Unhealthy for Sensitive Groups	🟧 Orange
151–200	Unhealthy	🟥 Red
201–300	Very Unhealthy	🟪 Purple
301–500	Hazardous	🟫 Brown
📍 API Requirements

You need a free API key from OpenWeather:
https://home.openweathermap.org/api_keys

Enable:

Air Pollution API

Current Weather Data API

🛠️ Project Structure
vehicle_aqi_project/
│
├── aqi_app_real_color.py       # Main Gradio app
├── vehicle_aqi_detection.py    # Initial YOLO detection experiments
├── test_app.py                 # Streamlit test (no longer used)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

🚀 Future Improvements

Deploy to HuggingFace Spaces or Streamlit Cloud

Switch to Random Forest or XGBoost for better AQI prediction

Real-time webcam-based AQI forecasting

Add GPS auto-location feature

🙌 Acknowledgements

Ultralytics YOLOv8 for object detection

Gradio for UI

OpenWeatherMap for AQI & weather APIs

📜 License

This project is open-source and free to use for educational and research purposes.
