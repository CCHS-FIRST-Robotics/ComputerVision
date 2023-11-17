import cv2
import numpy as np
import yaml


class Detector:
    def __init__(self, cfg):
        if cfg["marker_family"] == "16h5":
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_APRILTAG_16H5)
        else:
            raise ValueError("Unknown Family!")
        self.parameters = cv2.aruco.DetectorParameters()

    def detect(self, img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            grayscale_img, self.aruco_dict, None, parameters=self.parameters)
        if ids == None:
            return None
        return [corners, ids]


def distance_to_pixels(pixel_length, cfg):
    theta = np.radians(cfg["vert_fov"])
    theta_prime = pixel_length / cfg["height"] * theta
    return cfg["marker_size_in"] / np.tan(theta_prime)


if __name__ == '__main__':
    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    det = Detector(cfg)
    cap = cv2.VideoCapture(cfg["camera_id"])
    if not cap.isOpened():
        print("unable to open camera")
        quit(-1)
    cap.set(3, cfg["width"])
    cap.set(4, cfg["height"])

    while True:
        ret, img = cap.read()
        markers = det.detect(img)
        if markers:
            corners, ids = markers
            frame_markers = cv2.aruco.drawDetectedMarkers(
                img.copy(), corners, ids)
            for i in corners:
                (topLeft, topRight, bottomRight, bottomLeft) = i.reshape((4, 2))
                leftLength = topLeft[1] - bottomLeft[1]
                rightLength = topRight[1] - bottomRight[1]
                # bottomLength = bottomRight[0] - bottomLeft[0]
                # topLength = topRight[0] - topLeft[0]
                dist = distance_to_pixels(
                    (leftLength + rightLength) / 2, cfg)
                print(f"{dist} inches")
        cv2.imshow("April Tag Testing", img)
        if cv2.waitKey(1) == ord('q'):
            break
