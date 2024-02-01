import cv2
from cv2 import aruco
import numpy as np
import yaml

class Detector:

    def __init__(self,marker_family, marker_size, fovh, marker_ids):
        self.marker_size = marker_size
        self.fovh = fovh
        self.marker_ids = marker_ids
        if marker_family == "16h5":
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16H5)
        elif marker_family == "32h16":
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
        else:
            raise Exception("Unknown Family!")
        self.parameters = aruco.DetectorParameters()
    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
          self.aruco_dict, None, parameters=self.parameters)
        if not ids:
            return None
        #markers = [corners, ids]
        markers = []
#32h16

        for id, c in zip(ids, corners):
            if id not in self.marker_ids: continue


            (topLeft, topRight, bottomRight, bottomLeft) = c.reshape((4, 2))

            # topRight = (float(topRight[0]), float(topRight[1]))
            # bottomRight = (float(bottomRight[0]), float(bottomRight[1]))
            # bottomLeft = (float(bottomLeft[0]), float(bottomLeft[1]))
            # topLeft = (float(topLeft[0]), float(topLeft[1]))

            x = (topRight[0] + topLeft[0] + bottomRight[0] + bottomLeft[0]) / 4
            y = (topRight[1] + topLeft[1] + bottomRight[1] + bottomLeft[1]) / 4
            w1 = topRight[0] - topLeft[0]
            w2 = bottomRight[0] - bottomLeft[0]
            w = (w1 + w2) // 2
            w = int(w)
            x = int(x)
            y = int(y)
            




            # mark = {"center": centroid}
            mark = {"id": id, "x": x, "y":y}
            markers.append(mark)


        return markers












if __name__ == '__main__' :
    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    det = Detector(cfg["marker_family"])
    cap = cv2.VideoCapture(cfg["camera_id"])
    if not cap.isOpened():
        print("err")
        quit()

    cap.set(3, cfg["width"])
    cap.set(4, cfg["height"])
    while True:
        ret, img = cap.read()
        markers = det.detect(img)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(markers)
        # for m in markers:
        #     for c in m["corners":
        #
        if markers:
            corners, ids = markers
            # print(markers[0]['corners'])

            frame_markers = aruco.drawDetectedMarkers(img, corners, ids)

            (topLeft, topRight, bottomRight, bottomLeft) = corners[0].reshape((4, 2))

            topRight = (float(topRight[0]), float(topRight[1]))
            bottomRight = (float(bottomRight[0]), float(bottomRight[1]))
            bottomLeft = (float(bottomLeft[0]), float(bottomLeft[1]))
            topLeft = (float(topLeft[0]), float(topLeft[1]))

            centroid = det.centroid(topRight, topLeft, bottomRight, bottomLeft)


            # print(corners)
            # print(center)s
            cv2.rectangle(img, centroid, np.add(centroid, (4, 4)), (0, 0, 255), 2)
        # plt.savefig("_data/markers.pdf")
        # plt.show()

        cv2.imshow("markers", img)

        if cv2.waitKey(1) == ord('q'):
            break
   # [{"id":12, }]

