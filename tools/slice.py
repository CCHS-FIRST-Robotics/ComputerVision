import os

import cv2

img_dir = "images"
out_dir = "imgs_out"
cam_ix = 2

for row in os.scandir(img_dir):

    print(row.name)
    img = cv2.imread(row.path)
    w = img.shape[1] // 4
    fr = cam_ix * w
    to = (cam_ix + 1) * w
    img = img[:, fr:to, :]
    cv2.imwrite(f"{out_dir}/{row.name}", img)
    print(img.shape)
