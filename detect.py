from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Perform object detection on an image 
results = model("test/001.jpg")
results[0].show()

