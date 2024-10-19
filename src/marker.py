from multiprocessing import Semaphore, Value, shared_memory

import cv2
import numpy as np

from utils import get_dim, get_shm_frame


def marker_detect(cfg, shm, sem, procid, quit):

    cam = cfg["camera"]
    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])

    while True:
        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))
        # sem.acquire()
        # shm_array = np.ndarray(shape=(th, tw, cam["c"]), dtype=np.uint8, buffer=shm.buf)
        # frame = np.copy(shm_array)
        # sem.release()

        cv2.imshow("marker det " + str(procid), frame)

        if quit.value:
            break
        if cv2.waitKey(1) == 27:
            quit.value = 1
            break
