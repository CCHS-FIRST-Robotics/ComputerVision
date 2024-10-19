import time
from multiprocessing import Process, Semaphore, Value, shared_memory

import cv2
import numpy as np
import yaml

from arducam_utils import ArducamUtils
from marker import detect
from utils import fourcc, get_dim


def capture(cam, shm, sem, procid, quit):

    pixelformat = fourcc(cam["pformat"])

    cap = cv2.VideoCapture(cam["id"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, pixelformat)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["h"])

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    h = int(h)
    w = int(w)

    arducam_utils = ArducamUtils(cam["id"])
    cap.set(cv2.CAP_PROP_CONVERT_RGB, arducam_utils.convert2rgb)
    arducam_utils.write_dev(ArducamUtils.CHANNEL_SWITCH_REG, -1)

    # Turn off auto exposure
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # set exposure time
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)

    prev_tm = time.time()

    tw, th = get_dim(w, h, cam["wr"])

    while cap.isOpened():
        ret, frame = cap.read()
        sem.acquire()

        frame = frame.reshape(h, w)
        frame = arducam_utils.convert(frame)
        frame = cv2.resize(frame, (tw, th))

        shm_array = np.ndarray(shape=frame.shape, dtype=np.uint8, buffer=shm.buf)
        np.copyto(shm_array, frame)
        sem.release()

        cv2.imshow("video", frame)
        now = time.time()
        print(f"FPS {1/(now-prev_tm):.1f}")
        prev_tm = now

        if quit.value:
            break
        if cv2.waitKey(1) == 27:
            quit.value = 1
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    cam = cfg["camera"]

    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])
    shm_sz = tw * th * cam["c"]  # 3 channel BGR

    shm = shared_memory.SharedMemory(create=True, size=shm_sz)
    sem = Semaphore(1)
    quit = Value("i", 0)

    proc_cap = Process(target=capture, args=(cam, shm, sem, 0, quit))
    proc_marker = Process(target=detect, args=(cfg, shm, sem, 1, quit))

    proc_cap.start()
    proc_marker.start()

    proc_cap.join()
    proc_marker.join()

    shm.close()
    shm.unlink()
