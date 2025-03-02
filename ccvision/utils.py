import os

import numpy as np


def fourcc(fcc):
    n = 0
    for i in range(len(fcc)):
        f = ord(fcc[i]) << (i * 8)
        n |= f
    return n


def get_dim(w, h, dst_width):
    scale = dst_width * 1.0 / w
    return int(scale * w), int(scale * h)


def get_shm_frame(shm, sem, shape):
    sem.acquire()
    shm_array = np.ndarray(shape=shape, dtype=np.uint8, buffer=shm.buf)
    frame = np.copy(shm_array)
    sem.release()
    return frame


def rad2deg(rad):
    return 180 * rad / np.pi


def is_daemon():
    try:
        ppid = os.getpid()
        with open(f"/proc/{ppid}/comm", "r") as f:
            pp = f.read().strip()
            return pp == "systemd"

    except Exception:
        return False
