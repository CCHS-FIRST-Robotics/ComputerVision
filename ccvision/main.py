import argparse
import logging
import os
import signal
import time
from multiprocessing import Process, Semaphore, Value, shared_memory

import cv2
import numpy as np
import yaml

from .arducam_utils import ArducamUtils
from .marker import marker_detect
from .marker_single_cam import marker_detect_single
from .objdet import object_detect
from .stream import stream
from .utils import fourcc, get_dim, is_daemon

quit = Value("i", 0)


def signal_handler(sig, frame):
    print("Ctrl+C!", sig, frame)
    quit.value = 1
    # sys.exit(0)


def capture(cam, shm, sem, procid, quit):
    win_name = f"main {procid}"
    pixelformat = fourcc(cam["pformat"])

    cap = cv2.VideoCapture(cam["id"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, pixelformat)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["h"])

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    h = int(h)
    w = int(w)

    if cam["type"] == "arducam":
        arducam_utils = ArducamUtils(cam["id"])
        cap.set(cv2.CAP_PROP_CONVERT_RGB, arducam_utils.convert2rgb)
        arducam_utils.write_dev(ArducamUtils.CHANNEL_SWITCH_REG, -1)

    # set exposure time
    if cam['exposure'] == "auto":
        print("auto exposure")
        print("mm",cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        print("main",cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, cam["exposure"])

    p_tm = time.time()
    tw, th = get_dim(w, h, cam["wr"])

    logging.basicConfig(
        filename=cfg["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("capture start")
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        sem.acquire()

        frame = frame.reshape(h, w)

        if cam["type"] == "arducam":
            frame = arducam_utils.convert(frame)

        frame = cv2.resize(frame, (tw, th))

        shm_array = np.ndarray(shape=frame.shape, dtype=np.uint8, buffer=shm.buf)
        np.copyto(shm_array, frame)
        sem.release()

        if i % 30 == 0:
            logging.info(f"capture {i}")

        i += 1

        if cfg["display"]["main"]:
            now = time.time()
            fps = f"FPS {1/(now-p_tm):.1f}"
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

            logging.info("capture quit 1")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("capture quit")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launcher")
    parser.add_argument(
        "-c", "--config", type=str, default="config.yaml", help="config file"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    # Load config file
    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    # Configure logging
    logging.basicConfig(
        filename=cfg["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # turn off display when running as a daemon
    if is_daemon():
        cfg["is_daemon"] = True
        for t in cfg["display"]:
            cfg["display"][t] = False

    cam = cfg["camera"]

    # precalculate constant values
    cam["imw"] = cam["wr"] // 4  # one camera width

    # conv to rad
    cam["yaw_rad"] = [np.pi * a / 180 for a in cam["yaw"]]
    cam["pitch_rad"] = [np.pi * a / 180 for a in cam["pitch"]]

    # Load camera calibration
    if os.path.exists(cam["calibration"]):
        with np.load(cam["calibration"]) as cal:
            cam["mtx"], cam["dist"], _, _ = [
                cal[i] for i in ("mtx", "dist", "rvecs", "tvecs")
            ]
    else:
        cam["mtx"] = None
        cam["dist"] = None

    tw, th = get_dim(cam["w"], cam["h"], cam["wr"])
    shm_sz = tw * th * cam["c"]  # 3 channel BGR

    shm = shared_memory.SharedMemory(create=True, size=shm_sz)
    sem = Semaphore(1)

    proc_cap = Process(target=capture, args=(cam, shm, sem, 0, quit))
    proc_cap.start()

    if cfg["tasks"]["marker"]:
        proc_marker = Process(target=marker_detect, args=(cfg, shm, sem, 1, quit))
        proc_marker.start()

    if cfg["tasks"]["objdet"]:
        proc_objdet = Process(target=object_detect, args=(cfg, shm, sem, 2, quit))
        proc_objdet.start()

    if cfg["tasks"]["stream"]:
        proc_stream = Process(target=stream, args=(cfg, shm, sem, 3, quit))
        proc_stream.start()

    if cfg["tasks"]["marker_single"]:
        proc_marker_s = Process(target=marker_detect_single, args=(cfg, 4, quit))
        proc_marker_s.start()

    tasks = []
    for k, v in cfg["tasks"].items():
        if v:
            tasks.append(k)

    logging.info("ccvision starting with processes: " + " ".join(tasks))

    proc_cap.join()

    if cfg["tasks"]["marker"]:
        proc_marker.join()

    if cfg["tasks"]["objdet"]:
        proc_objdet.join()

    if cfg["tasks"]["stream"]:
        proc_stream.join()

    if cfg["tasks"]["marker_single"]:
        proc_marker_s.join()

    shm.close()
    shm.unlink()
