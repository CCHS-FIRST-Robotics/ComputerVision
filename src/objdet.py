import cv2

from utils import get_dim, get_shm_frame

# import numpy as np


def object_detect(cfg, shm, sem, procid, quit):

    cam = cfg["camera"]
    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])

    while True:
        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))
        cv2.imshow("obj det " + str(procid), frame)

        if quit.value:
            break
        if cv2.waitKey(1) == 27:
            quit.value = 1
            break
