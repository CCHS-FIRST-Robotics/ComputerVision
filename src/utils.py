import cv2


def fourcc(fcc):
    n = 0
    for i in range(len(fcc)):
        f = ord(fcc[i]) << (i * 8)
        n |= f
    return n


def resize(frame, dst_width):
    h = frame.shape[0]
    w = frame.shape[1]

    scale = dst_width * 1.0 / w
    return cv2.resize(frame, (int(scale * w), int(scale * h)))
