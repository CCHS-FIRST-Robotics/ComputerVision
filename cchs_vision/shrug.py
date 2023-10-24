import cv2
import aruco from cv2
import numpy as num
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd
# %matplotlib nbagg
def dist(x1, y1, x2, y2):
    return num.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
# Load an image 
cap = cv2.VideoCapture(0)
# img = cv2.imread("starry_night.jpg") 
# img = cap.read()
 
if not cap.isOpened():
    print("Cannot open camera")
    quit()
 
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16H5)
parameters =  aruco.DetectorParameters()



while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

    
    # F = P * D / W

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, None, parameters=parameters)
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
        

        
    frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imshow("The hell is numpy", frame_markers)
    # plt.savefig("_data/markers.pdf")
    # plt.show()

    if cv2.waitKey(1) == ord('q'):
            break    
