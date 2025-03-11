from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Perform object detection on an image 
results = model("test/004.jpg")
results[0].show()

