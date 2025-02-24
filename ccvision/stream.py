import subprocess
import time

import cv2
import numpy as np

from .utils import get_dim, get_shm_frame


def stream(cfg, shm, sem, procid, quit):
    win_name = f"stream det {procid}"

    cam = cfg["camera"]
    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])

    # Open a GStreamer pipe for H.265 encoding and streaming
    gst_output_pipeline = (
        f"appsrc ! "
        "videoconvert ! "
        "nvv4l2h265enc bitrate={cfg['stream']['bitrate']} ! "
        "rtph265pay ! "
        "udpsink host={cfg['stream']['ip']} port={cfg['stream']['port']}"
    )
    gst_process = subprocess.Popen(
        gst_output_pipeline, shell=True, stdin=subprocess.PIPE
    )

    p_tm = time.time()

    while True:

        frame = get_shm_frame(shm, sem, (th, tw, cam["c"]))  # hwc 400, 2560, 3

        # Stream only one camera specified in cfg default
        i = cfg['stream']['camid']    
        framei = frame[:, i * cam['imw'] : (i + 1) * cam['imw'], :]

        gst_process.stdin.write(framei.tobytes())
        if cv2.waitKey(1) == 27:
            gst_process.terminate()
            quit.value = 1
            break

        if quit.value: # from other process
            gst_process.terminate()
            break
