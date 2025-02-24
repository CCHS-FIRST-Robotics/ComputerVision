import time

import cv2
import numpy as np
from ultralytics import YOLO

from .utils import get_dim, get_shm_frame


def object_detect(cfg, shm, sem, procid, quit):
    win_name = f"obj det {procid}"

    cam = cfg["camera"]
    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])

    trt_model = YOLO("yolo11n.engine", task="detect")

    p_tm = time.time()

    while True:

        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))  # hwc 400, 2560, 3
        results = {}

        # Do object detection only cameras specified in cfg
        for i in cfg["objdet"]["cameraids"]:
            framei = frame[:, i * cam["imw"] : (i + 1) * cam["imw"], :]
            res = trt_model.predict(framei, verbose=False)
            results[i] = res

        if cfg["display"]["objdet"]:
            frames = []
            for i, res in results.items():
                for r in res:
                    frames.append(r.plot())

            iframe1 = np.hstack((frames[0], frames[1]))
            iframe2 = np.hstack((frames[2], frames[3]))
            iframe = np.vstack((iframe1, iframe2))

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
