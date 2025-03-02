import argparse
import math

import cv2
import numpy as np


def main(s, x, y):
    m = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    z = np.tile(m, (math.ceil(y / 2), math.ceil(x / 2)))
    z = z[:y, :x]
    z = np.repeat(z, s, axis=0)
    z = np.repeat(z, s, axis=1)
    z *= 255
    fn = f"chessboard_{x}_{y}.png"
    cv2.imwrite(fn, z)
    print(f"Saved {x}x{y} chessboard to {fn}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chessboard maker")
    parser.add_argument("-s", "--size", type=int, default=100, help="block size (px)")
    parser.add_argument("-x", "--xcount", type=int, default=9, help="x count")
    parser.add_argument("-y", "--ycount", type=int, default=7, help="y count")
    args = parser.parse_args()
    main(args.size, args.xcount, args.ycount)
