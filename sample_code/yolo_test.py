import cv2
import numpy as np
from ultralytics import YOLO # type: ignore

model = YOLO("yolo-Weights/yolov8n.pt")

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls = int(b.cls[0])
            conf = b.conf[0]
            txt = f"{r.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(img, txt, (x1, y1), font, fontScale, color, thickness)

            # print("b conf",b.conf[0])
            # print("cls",r.names[cls], b.conf[0])

    cv2.imshow("Hello", img)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()