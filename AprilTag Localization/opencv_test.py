import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while vid.isOpened():
    ret, frame = vid.read()

    if ret:
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()
    