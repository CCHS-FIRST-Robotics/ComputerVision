import logging
import time

import cv2
import numpy as np

from .angles import Angles
from .marker_detector import MarkerDetector
from .network_table import NetworkTable
from .utils import deg2rad, get_dim, get_shm_frame, put_fps


def marker_detect(cfg, quit):

    win_name = "marker det"
    cam = cfg["camera"]
    mark = cfg["marker"]

    # conv to rad
    cam["yaw_rad"] = deg2rad(cam["yaw"])
    cam["pitch_rad"] = deg2rad(cam["pitch"])

    cap = cv2.VideoCapture(cam["id"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["h"])

    if cam["exposure"] == "auto":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, cam["exposure"])

    print(
        "auto exposure",
        cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        cap.get(cv2.CAP_PROP_EXPOSURE),
    )

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    h = int(h)
    w = int(w)

    angles = Angles(w, h, cam["fovh"])
    detector = MarkerDetector(
        mark["family"], mark["size"], angles, cfg["display"]["marker"]
    )

    # Create a NetworkTables instance
    network = NetworkTable(cfg, "tags")
    packetid = 1000

    logging.basicConfig(
        filename=cfg["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("marker_detect single cam start")

    p_tm = time.time()
    while True:
        ret, frame = cap.read()
        markers = detector.detect(frame, cam["yaw_rad"], cam["pitch_rad"])

        if len(markers) > 0:
            markers.insert(0, packetid)
            network.send_array("tags", markers)
            packetid += 1

        if cfg["display"]["marker"]:
            now = time.time()
            fps = f"FPS {1/(now-p_tm):.1f}"
            p_tm = now

            put_fps(cfg, frame, fps)
            cv2.imshow(win_name, frame)

            if cv2.waitKey(1) == 27:
                quit.value = 1
                break

        if quit.value:
            logging.info("marker_detect single cam quit")
            break


def marker_detect4cam(cfg, shm, sem, quit):

    win_name = "marker det 4 cam"
    cam = cfg["camera4cam"]
    mark = cfg["marker"]

    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])

    angles = Angles(cam["imw"], th, cam["fovh"])
    detector = MarkerDetector(
        mark["family"], mark["size"], angles, cfg["display"]["marker4cam"]
    )

    # Create a NetworkTables instance
    network = NetworkTable(cfg, "tags")
    packetid = 0

    logging.basicConfig(
        filename=cfg["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("marker_detect4cam start")

    p_tm = time.time()
    cnt = 0
    while True:
        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))
        frames = []
        markers = []

        # Do marker detection only cameras specified in cfg
        for i in cam["cameraids"]:
            framei = frame[:, i * cam["imw"] : (i + 1) * cam["imw"], :]

            if cam["mtx"] is not None and cam["dist"] is not None:
                w = cam["imw"]
                # TODO preprocess outside loop
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                    cam["mtx"], cam["dist"], (w, th), 1, (w, th)
                )
                framei = cv2.undistort(
                    framei, cam["mtx"], cam["dist"], None, newcameramtx
                )
                x, y, w, h = roi
                framei = framei[y : y + h, x : x + w]
            frames.append(framei)
            mrkrs = detector.detect(framei, cam["yaw_rad"][i], cam["pitch_rad"][i])
            markers.extend(mrkrs)

        if len(markers) > 0:
            markers.insert(0, packetid)
            network.send_array("tags", markers)
            packetid += 1

        if cfg["display"]["marker4cam"]:
            iframe1 = np.hstack((frames[0], frames[1]))
            iframe2 = np.hstack((frames[2], frames[3]))
            iframe = np.vstack((iframe1, iframe2))

            now = time.time()
            fps = f"FPS {1/(now-p_tm):.1f}"
            p_tm = now

            put_fps(cfg, iframe, fps)
            cv2.imshow(win_name, iframe)

            if cv2.waitKey(1) == 27:
                quit.value = 1
                break

        if quit.value:
            logging.info("marker_detect4cam quit")
            break
