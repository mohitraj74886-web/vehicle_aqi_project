import cv2
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
model = YOLO("yolov8n.pt")

video_path = r"C:\Users\vvkra\Downloads\traffic.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

#fps allows you to convert frame numbers --> timestamp
all_detections = []      #list for storing the rows(frame,time,class etc)

frame_number = 0  #starting frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # YOLO detection

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]        # "car", "truck", etc.
        conf = float(box.conf[0])       # confidence score

        # Only count desired vehicle types
        allowed = ["car", "truck", "bus", "motorbike"]

        if label in allowed:
            time_sec = frame_number / fps  

            all_detections.append({
                "frame": frame_number,
                "time_sec": round(time_sec, 2),
                "class": label,
                "confidence": round(conf, 3)
            })

    frame_number += 1   #increase the frame by 1 and resume the loop

cap.release()

#explaination
#YOLO gives the bounding boxes --> we extract class and confidence from it
#only 4 types(cars,truck,bikes,bus)
#convert frame number --> timestamp
#store it in a list mentioned above

df = pd.DataFrame(all_detections)
df.to_csv("raw_detections.csv", index=False)

print("Saved raw detections to raw_detections.csv")
#this csv contains all detections with timestamps

counts_per_second = df.groupby("time_sec").size().reset_index(name="vehicle_count")  #for ml model
#groupby.size()=number of detections in each second

counts_per_second.to_csv("vehicle_counts_timeseries.csv", index=False)
print("Saved time-series to vehicle_counts_timeseries.csv")

plt.plot(counts_per_second["time_sec"], counts_per_second["vehicle_count"])
plt.xlabel("Time (seconds)")
plt.ylabel("Vehicle Count")
plt.title("Traffic Density Over Time")
plt.show()



