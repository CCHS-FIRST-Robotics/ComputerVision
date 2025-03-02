# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import argparse
import glob

import cv2
import numpy as np


def calibrate(img_dir, cal_file, chessboard=(7, 6)):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(f"{img_dir}/*.png")

    print("total", len(images))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grid = (7, 6)

        # Find the chess board corners
        # ret, corners = cv2.findChessboardCorners(gray, grid, cv2.CALIB_CB_ADAPTIVE_THRESH)
        ret, corners = cv2.findChessboardCorners(gray, grid)
        # flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER + cv2.CALIB_CB_NORMALIZE_IMAGE
        # ret, corners = cv2.findChessboardCornersSB(gray, grid, flags)

        if corners is not None:
            print(corners.shape)

        # If found, add object points, image points (after refining them)
        if ret:
            print("success", fname)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, grid, corners2, ret)
            cv2.imshow("img", img)
            cv2.waitKey()
        else:
            print("fail", fname)

    if len(objpoints) == 0:
        print("No objpoints")
        return

    print("Calibrate? y/n")
    k = cv2.waitKey()

    if k == ord("y"):
        retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        print(retval)
        print(mtx)
        print(dist)

        np.savez_compressed(cal_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

        print(f"saved {cal_file}")

        img = cv2.imread(images[0])
        h, w = img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]
        cv2.imshow("calibrated", dst)
        cv2.waitKey()

    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calibrate")
    parser.add_argument(
        "-d", "--dir", type=str, required=True, help="calibration image dir"
    )
    parser.add_argument(
        "-c",
        "--cal",
        type=str,
        default="calibration.npz",
        help="camera calibration matrices",
    )
    args = parser.parse_args()
    calibrate(args.dir, args.cal)

