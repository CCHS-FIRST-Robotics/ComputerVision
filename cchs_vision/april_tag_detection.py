import cv2
from cv2 import aruco
import numpy as np
import yaml

count = 0
class Detector:


    def __init__(self,cfg):
        if cfg["marker_family"] == "16h5":
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16H5)
        else:
            raise Exception("Unknown Family!")
        self.parameters = aruco.DetectorParameters()
    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
          self.aruco_dict, None, parameters=self.parameters)
        if not ids:
            return None
        markers = [corners, ids]
        #
        # markers = []
        #
        # for i, c in zip(ids, corners):
        #     #mark = {"id": i, "xrad":0.23, "yrad":0.1, "d":2}
        #     mark = {"id": i}
        #     markers.append(mark)
        #
        #     (topLeft, topRight, bottomRight, bottomLeft) = c.reshape((4, 2))
        #
        #     print(c.reshape((4, 2)))
        #
        #     topRight = (int(topRight[0]), int(topRight[1]))
        #     bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        #     bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        #     topLeft = (int(topLeft[0]), int(topLeft[1]))
        return markers










if __name__ == '__main__' :
    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    det = Detector(cfg)
    cap = cv2.VideoCapture(cfg["camera_id"])
    if not cap.isOpened():
        print("err")
        quit()

    cap.set(3, cfg["width"])
    cap.set(4, cfg["height"])
    while True:
        ret, img = cap.read()
        markers = det.detect(img)
        print(markers)
        # for m in markers:
        #     for c in m["corners":
        #
        if markers:
            corners, ids = markers
            frame_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        # plt.savefig("_data/markers.pdf")
        # plt.show()

        cv2.imshow("markers", img)



        if cv2.waitKey(1) == ord('q'):
            break
   # [{"id":12, }]

