import subprocess
import time

import cv2
import numpy as np

from utils import get_dim, get_shm_frame


def stream(cfg, shm, sem, procid, quit):
    win_name = f"stream det {procid}"

    cam = cfg["camera"]
    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])

    # the 4 cameras are combined into a wide image 400x2560
    imw = cfg["camera"]["wr"] // 4  # one camera width

    # Open a GStreamer pipe for H.265 encoding and streaming
    gst_output_pipeline = (
        "appsrc ! "
        "videoconvert ! "
        "nvv4l2h265enc bitrate=4000000 ! "
        "rtph265pay ! "
        "udpsink host=192.168.1.50 port=5000"  # Replace with receiver's IP and port
    )
    gst_process = subprocess.Popen(
        gst_output_pipeline, shell=True, stdin=subprocess.PIPE
    )

    p_tm = time.time()

    while True:

        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))  # hwc 400, 2560, 3
        results = {}

        # Stream only one camera specified in cfg default
        for i in cfg["objdet"]["cameraids"]:
            framei = frame[:, i * imw : (i + 1) * imw, :]

            now = time.time()
            fps = f"FPS {1/(now-p_tm):.1f}"
            p_tm = now

            framei = cv2.putText(
                framei,
                fps,
                cfg["FPS"]["org"],
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg["FPS"]["fontscale"],
                cfg["FPS"]["color"],
                cfg["FPS"]["thickness"],
                cv2.LINE_AA,
            )
            gst_process.stdin.write(framei.tobytes())
            if cv2.waitKey(1) == 27:
                gst_process.terminate()
                quit.value = 1
                break

        if quit.value:
            gst_process.terminate()
            break
