import cv2
import numpy as np
import time

from utils import get_dim, get_shm_frame


def marker_detect(cfg, shm, sem, procid, quit):

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 30)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    win_name = "marker det " + str(procid)
    
    cam = cfg["camera"]
    mark = cfg["marker"]

    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])

    if mark["family"] == "36h11":
        marker_dict = cv2.aruco.DICT_APRILTAG_36h11
    else:
        print("unknown marker", mark["family"] )
        return

    dictionary = cv2.aruco.getPredefinedDictionary(marker_dict)
    detectorparams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detectorparams)

    p_tm = time.time()
    while True:
        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))

        corners, markerids, rejects = detector.detectMarkers(frame)
        #markers = []
        cv2.aruco.drawDetectedMarkers(frame, corners, markerids)

        if markerids is not None:
            for c, id in zip(corners, markerids):
                c = c.squeeze()
                cx = int(c[:,0].sum() / 4)
                cy = int(c[:,1].sum() / 4)
                #markers.append({"id":id, "c":(cx, cy)})

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        now = time.time()
        fps = f"M FPS {1/(now-p_tm):.1f}"
        p_tm = now

        frame = cv2.putText(frame, fps, org, font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
 

        if quit.value:
            break
        if cv2.waitKey(1) == 27:
            quit.value = 1
            break
