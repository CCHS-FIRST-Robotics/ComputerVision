import cv2
import numpy as np

from .angles import Angles


class MarkerDetector:
    def __init__(self, family, marker_size, angles, display):
        self.display = display  # wether to display markers on image
        self.marker_size = marker_size
        if family == "36h11":
            marker_dict = cv2.aruco.DICT_APRILTAG_36h11
        else:
            raise "unknown marker " + family

        self.angles = angles
        dictionary = cv2.aruco.getPredefinedDictionary(marker_dict)
        detectorparams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, detectorparams)

    def detect(self, img, yaw_rad, pitch_rad):
        imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, markerids, rejects = self.detector.detectMarkers(imgg)
        markers = []
        # Calculate and draw center point
        if markerids is not None:
            for c, id in zip(corners, markerids):  # corners, markerid
                c = c.squeeze()
                cx = int(c[:, 0].sum() / 4)
                cy = int(c[:, 1].sum() / 4)

                pixel_sz1 = c[:, 0].max() - c[:, 0].min()
                pixel_sz2 = c[:, 1].max() - c[:, 1].min()
                pixel_sz = max(pixel_sz1, pixel_sz2)

                angleh_rad, anglev_rad = self.angles.get_angle(cx, cy)
                angleh_rad += yaw_rad

                if angleh_rad > np.pi:
                    angleh_rad -= 2 * np.pi

                anglev_rad += pitch_rad
                dist = self.angles.get_distance(pixel_sz, self.marker_size)
                markers.extend([id, angleh_rad, anglev_rad, dist])

                if self.display:
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

            if self.display:
                cv2.aruco.drawDetectedMarkers(img, corners, markerids)

        return markers
