# Import OpenCV (used to read video, show frames)
import cv2

# Import YOLO model from ultralytics
from ultralytics import YOLO

# 1. Load YOLO Model
# YOLOv8n is the "nano" version: smallest + fastest
model = YOLO("yolov8n.pt")
# This downloads the model automatically if missing


# 2. Load Video with OpenCV
video_path = r"C:\Users\vvkra\Downloads\traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened properly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 3. Read Video Frame-by-Frame
while True:
    ret, frame = cap.read()   # ret = True/False, frame = image array

    if not ret:               # If no more frames → exit loop
        break

    # 4. Run YOLO on the Frame
    results = model(frame)    # YOLO finds objects in the frame

    
    # 5. Draw YOLO Boxes on Frame
    
    annotated_frame = results[0].plot()
    # .plot() draws boxes + labels on the frame

    # 6. Display the Output Frame
    
    cv2.imshow("Vehicle Detection", annotated_frame)

    # Press ESC to quit
    if cv2.waitKey(1) == 27:
        break


# 7. Release Everything
cap.release()
cv2.destroyAllWindows()
