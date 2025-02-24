import os

import cv2


def calibrate(imgdir, camid):
    board_size = (9,6)
    frame_size = (640, 400)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    points = []
    
    for entry in os.scandir(imgdir):
        img = cv2.imread(entry.path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        if ret:
            #points.append(
            #imgpoints.append(corners)
            coerners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), crit)
            cv2.drawChessboardCorners(img, boardsize, corners, ret)
            cv2.imshow("img", img)
            cv2.waitKey(1)
    
    cv2.destroyAllWindows()

    ret, cam_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame


if __name__ == "__main__":
    calibrate(imgdir, camid)
