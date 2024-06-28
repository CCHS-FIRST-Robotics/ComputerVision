from typing import Any
import cv2
import cv2.typing as cvt
import numpy.typing as npt
import numpy as np

from classes.pose import Pose


class AprilTag:
    # Positions of each tag id in the world frame
    tag_poses = {
        # 1: Pose(0.2459,  1.3559, 15.0794, 0, 120 * np.pi/180, 0),
        # 2: Pose(0.8837,  1.3559, 16.1851, 0, 120 * np.pi/180, 0),
        3: Pose(4.9827, 1.4511, 16.5793, 0, 180 * np.pi / 180, 0),
        4: Pose(5.5479, 1.4511, 16.5793, 0, 180 * np.pi / 180, 0),
        # 5: Pose(8.2042,  1.3559, 14.7008, 0, 270 * np.pi/180, 0),
        # 6: Pose(8.2042,  1.3559,  1.8415, 0, 270 * np.pi/180, 0),
        7: Pose(5.5478, 1.4511, -0.0380, 0, 0, 0),
        8: Pose(4.9827, 1.4511, -0.0380, 0, 0, 0),
        # 9: Pose(0.8836,  1.3559,  0.3561, 0,  60 * np.pi/180, 0),
        # 10: Pose(0.2459, 1.3559,  1.4612, 0,  60 * np.pi/180, 0),
        # 11: Pose(3.7132, 1.3208, 11.9047, 0, 300 * np.pi/180, 0),
        # 12: Pose(4.4983, 1.3208, 11.9047, 0,  60 * np.pi/180, 0),
        13: Pose(4.1051, 1.3208, 11.2202, 0, 180 * np.pi / 180, 0),
        14: Pose(4.1051, 1.3208, 5.3208, 0, 0, 0),
        # 15: Pose(4.4983, 1.3208,  4.6413, 0, 120 * np.pi/180, 0),
        # 16: Pose(3.7132, 1.3208,  4.4613, 0, 240 * np.pi/180, 0)
    }

    def __init__(
        self, tag_id: int, tag_size: float, tag_family: str, tag_corners: npt.NDArray
    ) -> None:
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
        x_sum = (
            self.corners[0][0]
            + self.corners[1][0]
            + self.corners[2][0]
            + self.corners[3][0]
        )
        y_sum = (
            self.corners[0][1]
            + self.corners[1][1]
            + self.corners[2][1]
            + self.corners[3][1]
        )
        # Divide by 4 to get the center (this process is basically taking the average pixel value of the corners)
        self.center = (round(x_sum / 4), round(y_sum / 4))

    def draw_tag(self, image: cvt.MatLike, color=(0, 0, 255)) -> cvt.MatLike:
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

        color_inverted = (255 - color[0], 255 - color[1], 255 - color[2])

        cv2.circle(image, (center[0], center[1]), 5, color_inverted, 2)

        cv2.line(
            image, (corner_01[0], corner_01[1]), (corner_02[0], corner_02[1]), color, 2
        )
        cv2.line(
            image, (corner_02[0], corner_02[1]), (corner_03[0], corner_03[1]), color, 2
        )
        cv2.line(
            image, (corner_03[0], corner_03[1]), (corner_04[0], corner_04[1]), color, 2
        )
        cv2.line(
            image, (corner_04[0], corner_04[1]), (corner_01[0], corner_01[1]), color, 2
        )

        cv2.putText(
            image,
            str(self.id),
            (center[0] - 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color_inverted,
            2,
            cv2.LINE_AA,
        )
        # cv2.putText(image, f"({round(center[0])}, {round(center[1])})", (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        return image

    def get_world_translation(self) -> npt.NDArray[np.float32]:
        """Gets the translation of the AprilTag in the world frame

        Returns:
            npt.NDArray[np.float32]: (3,) translation vector
        """
        if self.id not in self.tag_poses.keys():
            return np.array([-1, -1, -1])
        return self.tag_poses[self.id].get_translation()

    def get_world_transformation(self) -> npt.NDArray[np.float32]:
        if self.id not in self.tag_poses.keys():
            return np.array([-1, -1, -1])
        return self.tag_poses[self.id].get_transformation_matrix()

    def get_corner_translations(self) -> npt.NDArray[np.float32]:
        """Gets the translation of each corner of the AprilTag in the world frame

        Returns:
            npt.NDArray[np.float32]: (4, 3) array of corner translations
        """
        tag_corners = np.array(
            [
                [-self.size / 2, self.size / 2, 0],
                [self.size / 2, self.size / 2, 0],
                [self.size / 2, -self.size / 2, 0],
                [-self.size / 2, -self.size / 2, 0],
            ]
        )
        # print(tag_corners + self.get_world_translation())
        # print()
        return tag_corners + self.get_world_translation()
