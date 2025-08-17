from ultralytics import YOLO
import numpy

model = YOLO("yolov8n.pt", "v11")

detection_output = model.predict(source = r"C:\Users\HP\Downloads\images (3).jpeg", conf = 0.25, save = True)

print(detection_output)

print(detection_output[0].numpy())