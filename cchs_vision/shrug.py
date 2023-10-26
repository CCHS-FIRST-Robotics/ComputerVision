import cv2
import numpy as np

def dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Load an image 
cap = cv2.VideoCapture(0)

# Quit if image cant be opened
if not cap.isOpened():
    print("err")
    quit()
 
# Get the family and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
parameters = cv2.aruco.DetectorParameters()



while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

    
    # F = P * D / W
    # Get corners and lengths
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, None, parameters=parameters)
    for i in corners:
        (topLeft, topRight, bottomRight, bottomLeft) = i.reshape((4, 2))
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        leftLength = topLeft[1] - bottomLeft[1]
        rightLength = topRight[1] - bottomRight[1]
        bottomLength = bottomRight[0] - bottomLeft[0]
        topLength = topRight[0] - topLeft[0]
        

        
    frame_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imshow("The hell is numpy", frame_markers)

    # Quit  on q
    if cv2.waitKey(1) == ord('q'):
            break    
