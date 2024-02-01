import numpy as np
import cv2 as cv
import glob
import pickle
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('cchs_vision2/imgcal/*.png')
for fname in images:
    img = cv.imread(fname)
    print(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1)

cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)
file_path = "camera_calib_matrices.txt"

data = {"mtx":mtx, "dist":dist}
cv.SOLVEPNP_IPPE_SQUARE = [[-0.0762, 0.0762, 0],[0.0762,0.0762,0],[0.0762,-0.0762,0],[-0.0762,-0.0762,0]]

print(mtx)
#Open the file in binary mode
#with open(file_path, 'w') as file:
    # Serialize and write the variable to the file
 #   pickle.dump(data, file)

