from ultralytics import YOLO

model = YOLO('YOLOv8n.pt')
results = model.train(data='FRC2024/data.yaml', epochs=5)

metrics = results.val()