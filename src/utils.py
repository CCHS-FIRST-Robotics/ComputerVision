import cv2


def fourcc(fcc):
    n = 0
    for i in range(len(fcc)):
        f = ord(fcc[i]) << (i * 8)
        n |= f
    return n


def get_dim(w, h, dst_width):
    scale = dst_width * 1.0 / w
    return int(scale * w), int(scale * h)
