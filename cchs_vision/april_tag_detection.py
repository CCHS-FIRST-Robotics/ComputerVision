import cv2
from cv2 import aruco
import numpy as np
import yaml
import pickle

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    with open(cfg["cam_cal"], "rb") as f:
        data = pickle.load(f)
        mtx = data["mtx"]
        dist = data["dist"]

    cap = cv2.VideoCapture(cfg["camera_id"])

    if not cap.isOpened():
        print("Error camera cannot open")
        quit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cfg["width"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cfg["height"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            continue

        cv2.imshow("markers", img)

        if cv2.waitKey(1) == ord("q"):
            break
