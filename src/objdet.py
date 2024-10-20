import time

import cv2

from utils import get_dim, get_shm_frame


def object_detect(cfg, shm, sem, procid, quit):
    win_name = f"obj det {procid}"

    cam = cfg["camera"]
    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])
    p_tm = time.time()

    while True:
        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))

        if cfg["display"]["objdet"]:

            cv2.imshow("obj det " + str(procid), frame)

            now = time.time()
            fps = f"M FPS {1/(now-p_tm):.1f}"
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
            break
