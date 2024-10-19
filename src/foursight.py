import time
from multiprocessing import Process, Semaphore, Value, shared_memory

import cv2
import yaml

from arducam_utils import ArducamUtils
from utils import fourcc, resize


def capture(shm, sem, cam, quit):

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

    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame.reshape(h, w)
        frame = arducam_utils.convert(frame)

        frame = resize(frame, 2560.0)

        cv2.imshow("video", frame)
        now = time.time()
        print(f"FPS {1/(now-prev_tm):.1f}")
        prev_tm = now

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    print(cfg)

    cam = cfg["camera"]

    shm_sz = cam["w"] * cam["h"] * cam["c"]  # 3 channel BGR

    print(cfg)
    print(shm_sz)

    capture("SHM", "sem", cam, "quit")
