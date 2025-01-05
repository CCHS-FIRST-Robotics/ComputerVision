from ultralytics import YOLO

# Load the exported TensorRT model
trt_model = YOLO("yolo11n.engine")

# Run inference
results = trt_model("dog.jpg")

for r in results:
    r.show()
