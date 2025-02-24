import cv2
import os

img_dir = "images"
out_dir = "imgs_out"

for row in os.scandir(img_dir):

    print(row.name)
    img = cv2.imread(row.path)
    w = img.shape[1] // 4
    img = img[:,:w,:]
    cv2.imwrite(f"{out_dir}/{row.name}", img)
    print(img.shape)
