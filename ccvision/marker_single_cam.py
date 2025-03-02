import logging
import time

import cv2
import numpy as np

from .angles import Angles
from .network_table import NetworkTable
from .utils import get_dim


def marker_detect_single(cfg, procid, quit):

    logging.basicConfig(
        filename=cfg["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"marker detect single pid {procid}")

    win_name = f"marker det {procid}"
    cam = cfg["camera_single"]
    mark = cfg["marker"]

    # conv to rad
    cam["yaw_rad"] = np.pi * cam["yaw"] / 180
    cam["pitch_rad"] = np.pi * cam["pitch"] / 180

    if mark["family"] == "36h11":
        marker_dict = cv2.aruco.DICT_APRILTAG_36h11
    else:
        print("unknown marker", mark["family"])
        return

    dictionary = cv2.aruco.getPredefinedDictionary(marker_dict)
    detectorparams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detectorparams)

    cap = cv2.VideoCapture(cam["id"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["h"])
    cap.set(cv2.CAP_PROP_EXPOSURE, cam["exposure"])

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    h = int(h)
    w = int(w)

    angles = Angles(w, h, cam["fovh"])

    # Create a NetworkTables instance
    network = NetworkTable(cfg, "tags")
    packetid = 1000

    i = 0
    p_tm = time.time()
    while True:

        markers = []

        ret, frame = cap.read()
        frameig = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, markerids, rejects = detector.detectMarkers(frameig)

        if i % 10 == 0:
            logging.info(f"mark single {i}")

        i += 1

        # Calculate and draw center point
        if markerids is not None:
            for c, id in zip(corners, markerids):  # corners, markerid
                c = c.squeeze()
                cx = int(c[:, 0].sum() / 4)
                cy = int(c[:, 1].sum() / 4)

                # TODO change to use sides
                pixel_sz1 = c[:, 0].max() - c[:, 0].min()
                pixel_sz2 = c[:, 1].max() - c[:, 1].min()
                pixel_sz = max(pixel_sz1, pixel_sz2)

                angleh_rad, anglev_rad = angles.get_angle(cx, cy)
                angleh_rad += cam["yaw_rad"]

                if angleh_rad > np.pi:
                    angleh_rad -= 2 * np.pi

                anglev_rad += cam["pitch_rad"]

                dist = angles.get_distance(pixel_sz, cfg["marker"]["size"])

                markers.extend([id, angleh_rad, anglev_rad, dist])

                if cfg["display"]["marker"]:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        if len(markers) > 0:
            markers.insert(0, packetid)
            network.send_array("tags", markers)
            packetid += 1

        if cfg["display"]["marker_single"]:
            cv2.aruco.drawDetectedMarkers(frame, corners, markerids)

            now = time.time()
            fps = f"FPS {1/(now-p_tm):.1f}"
            p_tm = now

            frame = cv2.putText(
                frame,
                fps,
                cfg["FPS"]["org"],
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg["FPS"]["fontscale"],
                cfg["FPS"]["color"],
                cfg["FPS"]["thickness"],
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, frame)

            if cv2.waitKey(1) == 27:
                quit.value = 1
                break

        if quit.value:
            logging.info(f"marker detect single quit")
            break
