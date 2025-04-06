import cv2
import numpy as np
from ultralytics import YOLO

#model = YOLO("yolo-Weights/yolov8n.pt")
model = YOLO("yolo11n.pt")

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 255, 0)
thickness = 2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0] #coordinates of bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) #convert to int coordinates

            classID = int(b.cls[0])
            confidence = b.conf[0]

            txt = f"{r.names[classID]} {confidence:.2f}" #what is detected, & confidence rating

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(img, txt, (x1, y1), font, fontScale, color, thickness)

    cv2.imshow("frame", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
