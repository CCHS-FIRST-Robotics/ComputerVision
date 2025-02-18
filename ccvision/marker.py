import time

import cv2
import numpy as np

from .angles import Angles
from .NetworkTable import NetworkTable
from .utils import get_dim, get_shm_frame


def marker_detect(cfg, shm, sem, procid, quit):

    win_name = f"marker det {procid}"
    cam = cfg["camera"]
    mark = cfg["marker"]

    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])

    if mark["family"] == "36h11":
        marker_dict = cv2.aruco.DICT_APRILTAG_36h11
    else:
        print("unknown marker", mark["family"])
        return

    dictionary = cv2.aruco.getPredefinedDictionary(marker_dict)
    detectorparams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detectorparams)

    angles = Angles(cam["imw"], th, cam["fovh"])

    # Create a NetworkTables instance
    network = NetworkTable(cfg, "tags")
    packetid = 0

    p_tm = time.time()
    while True:
        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))
        frames = {}
        markers = []

        # Do imarker detection only cameras specified in cfg
        for i in cfg["marker"]["cameraids"]:
            framei = frame[:, i * cam["imw"] : (i + 1) * cam["imw"], :]
            framei = cv2.cvtColor(framei, cv2.COLOR_RGB2GRAY)
            corners, markerids, rejects = detector.detectMarkers(framei)

            # Calculate and draw center point
            if markerids is not None:
                for c, id in zip(corners, markerids):  # corners, markerid
                    c = c.squeeze()
                    cx = int(c[:, 0].sum() / 4)
                    cy = int(c[:, 1].sum() / 4)

                    # TODO change to use sides
                    pixel_sz = c[:, 0].max() - c[:, 0].min()

                    angleh_rad, anglev_rad = angles.get_angle(cx, cy)
                    dist = angles.get_distance(pixel_sz, cfg["marker"]["size"])

                    pose = 0
                    markers.extend([id, pose, angleh_rad, anglev_rad, dist])

                    if cfg["display"]["marker"]:
                        cv2.circle(framei, (cx, cy), 4, (0, 0, 255), -1)

            if cfg["display"]["marker"]:
                cv2.aruco.drawDetectedMarkers(framei, corners, markerids)
                frames[i] = framei

        if len(markers) > 0:
            network.send_array("tags", markers.insert(0, packetid))
            packetid += 1

        if cfg["display"]["marker"]:
            iframe = np.hstack(list(frames.values()))

            now = time.time()
            fps = f"FPS {1/(now-p_tm):.1f}"
            p_tm = now

            frame = cv2.putText(
                iframe,
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
            break
