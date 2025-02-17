import numpy as np


class Angles:
    def __init__(self, img_w, img_h, fovh):
        self.img_h = img_h
        self.img_w = img_w

        fovh_rad = np.pi * fovh / 180
        # calculate vertical camera angle with square pixels
        self.fovv_rad = 2 * np.arctan(img_h / img_w * np.tan(fovh_rad / 2))

        # focal length in pixels
        self.focal_px = img_w / 2 / np.tan(fovh_rad / 2)

    def get_angle(self, x, y):
        """
        pixels are from top left
        angles measured from camera center
        """
        x -= self.img_w / 2  # center right positive
        y = self.img_h / 2 - y  # y start from top, reverse

        angleh_rad = np.arctan(x / self.focal_px)
        anglev_rad = np.arctan(y / self.focal_px)

        return angleh_rad, anglev_rad

    def get_distance(self, pixel_sz, marker_sz):
        d = marker_sz * self.focal_px / pixel_sz
        return d
