import cv2
# from cv2 import aruco
import numpy as np
import yaml
import pickle
import numpy


def centroid(topRight, topLeft, bottomRight, bottomLeft):
    # Convert to points into points with floats for float division
    # Can comment out if it already float
    topRight = (float(topRight[0]), float(topRight[1]))
    bottomRight = (float(bottomRight[0]), float(bottomRight[1]))
    bottomLeft = (float(bottomLeft[0]), float(bottomLeft[1]))
    topLeft = (float(topLeft[0]), float(topLeft[1]))

    # Gets slope between 2 points, rise over run
    m1 = (bottomLeft[1] - topRight[1]) / (bottomLeft[0] - topRight[0])
    m2 = (topLeft[1] - bottomRight[1]) / (topLeft[0] - bottomRight[0])

    # Calculate b
    # y = mx + b
    # y - mx = b
    b1 = topRight[1] - m1 * topRight[0]
    b2 = topLeft[1] - m2 * topLeft[0]

    # Calculate system, find intersection of
    # y = m1 x + b1
    # y = m2 x + b2
    # Set equal, m2 x + b2 = m1 x + b1
    # Transfer, x (m2 - m1) = b1 - b2
    # Divide, x = (b1 - b2) / (m2 - m1)
    x = (b1 - b2) / (m2 - m1)
    # Y is found by plugging it back in
    y = m1 * x + b1
    return (x, y)


def slope(point1, point2):
    return (point1[1] - point2[1]) / (point1[0] - point2[0])


class pnp_detect:
    def __init__(self, mtx, dist) -> None:
        self.mtx = mtx
        self.dist = dist
        self.ob_points = cv2.SOLVEPNP_IPPE_SQUARE = np.array(
            [
                [-6 / 2, 6 / 2, 0.0],
                [6 / 2, 6 / 2, 0.0],
                [6 / 2, -6 / 2, 0.0],
                [-6 / 2, -6 / 2, 0.0],
            ],
            dtype=np.float64,
        )

    def detect(self, places):
        if places is None:
            return None, None
        self.fake_points = np.array(
            [
                [places["topRight[0]"], places["topRight[1]"]],
                [places["topLeft[0]"], places["topLeft[1]"]],
                [places["bottomLeft[0]"], places["bottomLeft[1]"]],
                [places["bottomRight[0]"], places["bottomRight[1]"]],
            ],
            dtype=np.float32,
        )

        _, rvecs, tvecs = cv2.solvePnP(
            self.ob_points, self.fake_points, self.mtx, self.dist
        )
        return rvecs, tvecs


class Detector:
    def __init__(self, cfg, mtx, dist):
        self.cfg = cfg

        self.tan_half_fovh = np.tan(np.radians(cfg["fovh"]) / 2)

        self.fovv = np.arctan(
            cfg["height"] * (np.tan(np.radians(cfg["fovh"])) / cfg["width"])
        )
        self.tan_half_fovv = np.tan(self.fovv / 2)

        self.half_width = self.cfg["width"] / 2
        self.half_height = self.cfg["height"] / 2

        self.tan_half_fovh_half_width = self.tan_half_fovh / self.half_width
        self.tan_half_fovv_half_width = self.tan_half_fovv / self.half_width

        self.tan_half_fovh_half_height = self.tan_half_fovh / self.half_height
        self.tan_half_fovv_half_height = self.tan_half_fovv / self.half_height

        self.wcoef = self.cfg["marker_size"] / self.tan_half_fovh_half_width
        self.hcoef = self.cfg["marker_size"] / self.tan_half_fovv_half_width

        if cfg["marker_family"] == "16h5":
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        elif cfg["marker_family"] == "36h11":
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        else:
            raise Exception("Unknown Family!")
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters) 
        self.mtx = mtx
        self.dist = dist

    def detect(self, img):
        cfg = self.cfg
        img = cv2.undistort(img, self.mtx, self.dist)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        

        if ids is None:
            return None, None

        markers = []

        rvecs, tvecs = 0, 0

        for id, c in zip(ids, corners):
            id = id[0]
            if id not in self.cfg["marker_ids"]:
                continue

            # get width of April Tag in pixels
            (topLeft, topRight, bottomRight, bottomLeft) = np.squeeze(c)
            # print((topLeft, topRight, bottomRight, bottomLeft))
            leftLength = topLeft[1] - bottomLeft[1]
            rightLength = topRight[1] - bottomRight[1]
            bottomLength = bottomRight[0] - bottomLeft[0]
            topLength = topRight[0] - topLeft[0]

            pixel_width = (bottomLength + topLength) / 2

            # explained in docs/distance_formula.md
            w_distance = self.wcoef / pixel_width

            x = (topRight[0] + topLeft[0] + bottomRight[0] + bottomLeft[0]) / 4
            y = (topRight[1] + topLeft[1] + bottomRight[1] + bottomLeft[1]) / 4
            w1 = topRight[0] - topLeft[0]
            w2 = bottomRight[0] - bottomLeft[0]
            w = (w1 + w2) // 2
            w = int(w)

            (center_x, center_y) = centroid(topRight, topLeft, bottomRight, bottomLeft)
            center_x -= cfg["width"] / 2
            center_y -= cfg["height"] / 2

            # TODO: change pixel_width and pixel_height to distance axes
            hangle = np.arctan(center_x * self.tan_half_fovh_half_width)
            vangle = np.arctan(center_y * self.tan_half_fovv_half_width)

            # mark = {"center": centroid}
            # mark = {"id": id, "dis": min((np.abs(h_distance), np.abs(w_distance))), "hang": hangle, "vang": vangle}
            mark = {"id": id, "dis": np.abs(w_distance), "hang": hangle, "vang": vangle}
            mark_data = {
                "topLeft[0]": topLeft[0],
                "topRight[0]": topRight[0],
                "bottomRight[0]": bottomRight[0],
                "bottomLeft[0]": bottomLeft[0],
                "topLeft[1]": topLeft[1],
                "topRight[1]": topRight[1],
                "bottomRight[1]": bottomRight[1],
                "bottomLeft[1]": bottomLeft[1],
            }
            markers.append(mark)

        return markers


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    with open(cfg["cam_cal"], "rb") as f:
        data = pickle.load(f)
        mtx = data["mtx"]
        dist = data["dist"]

    cap = cv2.VideoCapture(cfg["camera_id"])

    if not cap.isOpened():
        print("err")
        quit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cfg["width"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cfg["height"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    det = Detector(cfg, mtx, dist)

    while cap.isOpened():
        ret, img = cap.read()
        # ret, places = det.detect(img)

        if not ret:
            continue
        ####
        # markers, rvecs, tvecs = ret
        # print(f"{rvecs=}, {tvecs=}")
        # det_pnp = pnp_detect(mtx, dist)
        # det_pnp.detect(places)

        # for marker in markers:
        #     cv2.putText(
        #         img,
        #         f"d={marker['dis']:.2f}",
        #         (0, 30),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         2,
        #         cv2.LINE_AA,
        #     )
        #     cv2.putText(
        #         img,
        #         f"h={np.degrees(marker['hang']):.2f}",
        #         (0, 60),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         2,
        #         cv2.LINE_AA,
        #     )
        #     cv2.putText(
        #         img,
        #         f"v={np.degrees(marker['vang']):.2f}",
        #         (0, 90),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         2,
        #         cv2.LINE_AA,
        #     )
            # print(f'Id: {marker["id"]} found at {marker["dis"]:.2f} metres away')
        # plt.savefig("_data/markers.pdf")
        # plt.show()

        cv2.imshow("markers", img)

        if cv2.waitKey(1) == ord("q"):
            break
    # [{"id":12, }]
