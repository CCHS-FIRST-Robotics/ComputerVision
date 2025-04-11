import cv2
import numpy as np
from cv2 import aruco
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    marker_dict_type = cv2.aruco.DICT_APRILTAG_36h11

    dictionary = aruco.getPredefinedDictionary(marker_dict_type)

    parameters = cv2.aruco.DetectorParameters()

    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()