import argparse
import os

import cv2
import yaml

from ccvision.arducam_utils import ArducamUtils
from ccvision.utils import fourcc, get_dim


def capture(cam, imgdir):
    win_name = "Capture"
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
    # set exposure time
    #cap.set(cv2.CAP_PROP_EXPOSURE, cam['exposure'])

    tw, th = get_dim(w, h, cam["wr"])
    print("Press s to save image. ESC to quit")
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        frame = frame.reshape(h, w)
        frame = arducam_utils.convert(frame)
        #frame = cv2.resize(frame, (tw, th))

        cv2.imshow(win_name, frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
        elif k == ord("s"):
            fn = f"{imgdir}/img_{i}.png"
            cv2.imwrite(fn, frame)
            print("saved", fn)
            i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launcher")
    parser.add_argument(
        "-c", "--config", type=str, default="config.yaml", help="config file"
    )
    parser.add_argument("-d", "--dir", type=str, default="images", help="image dir")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    cam = cfg["camera"]

    os.makedirs(args.dir, exist_ok=True)
    capture(cam, args.dir)
