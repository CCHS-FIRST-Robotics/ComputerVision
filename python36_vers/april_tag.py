from typing import Any
import cv2
import cv2.typing as cvt
#import numpy.typing as npt
import numpy as np

from pose import Pose

class AprilTag:
     
    # Positions of each tag id in the world frame
    tag_poses = {
        0: Pose(0, 0, 0, 0, 0, 0),
        1: Pose(1, 0, 0, 0, 0, 0),
        2: Pose(2, 0, 0, 0, 0, 0),
        3: Pose(3, 0, 0, 0, 0, 0),
    }
    
    def __init__(self, tag_id: int, tag_size: float, tag_family: str, tag_corners) -> None:
        """Initializes an AprilTag observation

        Args:
            tag_id (int): Id of the AprilTag
            tag_size (float): Size (length) of the AprilTag (meters)
            tag_family (str): Tag family of the AprilTag (i.e., '16h5')
            tag_corners (npt.NDArray): Array of the 4 corners of the detected AprilTag in pixel coordinates
        """
        self.id = tag_id
        self.size = tag_size
        self.family = tag_family
        self.corners = tag_corners
        
        # Sum up the x and y pixel values of each corner 
        x_sum = self.corners[0][0] + self.corners[1][0] + self.corners[2][0] + self.corners[3][0]
        y_sum = self.corners[0][1] + self.corners[1][1] + self.corners[2][1] + self.corners[3][1]
        # Divide by 4 to get the center (this process is basically taking the average pixel value of the corners)
        self.center = (round(x_sum/4), round(y_sum/4))
        
    def draw_tag(self, image: cvt.MatLike) -> cvt.MatLike:
        """Draws the AprilTag on an image

        Args:
            image (cvt.MatLike): image to draw the AprilTag on (should be a cv2 image)

        Returns:
            cvt.MatLike: annotated image
        """
        center = (int(self.center[0]), int(self.center[1]))
        corner_01 = (int(self.corners[0][0]), int(self.corners[0][1]))
        corner_02 = (int(self.corners[1][0]), int(self.corners[1][1]))
        corner_03 = (int(self.corners[2][0]), int(self.corners[2][1]))
        corner_04 = (int(self.corners[3][0]), int(self.corners[3][1]))

        cv2.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv2.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv2.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv2.putText(image, str(self.id), (center[0] - 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        #cv2.putText(image, f"({round(center[0])}, {round(center[1])})", (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        return image
    
    def get_world_translation(self):
        """Gets the translation of the AprilTag in the world frame

        Returns:
            npt.NDArray[np.float32]: (3,) translation vector
        """
        return self.tag_poses[self.id].get_translation()

    def get_corner_translations(self):
        """Gets the translation of each corner of the AprilTag in the world frame

        Returns:
            npt.NDArray[np.float32]: (4, 3) array of corner translations
        """
        tag_corners = np.array([
            [-self.size/2, self.size/2, 0],
            [self.size/2, self.size/2, 0],
            [self.size/2, -self.size/2, 0],
            [-self.size/2, -self.size/2, 0]
        ])
        # print(tag_corners + self.get_world_translation())
        # print()
        return tag_corners + self.get_world_translation()
