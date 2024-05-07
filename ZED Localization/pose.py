from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation #type: ignore

class Pose:
    
    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        """Initializes the pose from the x, y, z, yaw, pitch, and roll values

        Args:
            x (float): x component of the pose translation (meters)
            y (float): y component of the pose translation (meters)
            z (float): z component of the pose translation (meters)
            roll (float): roll (rotation about the x-axis) of the pose (radians)
            pitch (float): pitch (rotation about the y-axis) of the pose (radians)
            yaw (float): yaw (rotation about the z-axis) of the pose (radians)
        """
        self.x = x
        # print(x)
        self.y = y
        self.z = z
        
        self.rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        
    def __str__(self) -> str:
        """Converts the Pose object into a string (x, z, pitch) in inches/degrees

        Returns:
            str: string representation of the Pose object
        """
        return f"Pose:  x: {np.round(self.get_x_inches(), 3)}  y: {np.round(self.get_y_inches(), 3)}  z: {np.round(self.get_z_inches(), 3)}  depth: {np.round(self.get_depth_inches(), 2)}  pitch: {np.round(self.get_heading_degrees(), 3)}"
    
    def get_2d_pose(self) -> npt.NDArray[np.float32]:
        """Gets the 2d pose (x, z, pitch) of the pose

        Returns:
            npt.NDArray[np.float32]: (3,) 2d pose
        """
        return np.array([self.x, self.z, self.get_heading_degrees()])
    
    def get_3d_pose(self) -> npt.NDArray[np.float32]:
        """Gets the 3d pose (x, y, z, roll, pitch, yaw) of the pose

        Returns:
            npt.NDArray[np.float32]: (6,) 3d pose
        """
        return np.array([self.x, self.y, self.z, *Rotation.as_euler(self.rotation, 'xyz')])
    
    def get_as_wpi_pose(self) -> Pose:
        """Gets the pose as a WPILib pose (z, x, y, yaw, roll, pitch)

        Returns:
            npt.NDArray[np.float32]: (6,) WPILib pose
        """
        return Pose(self.z, self.x, self.y, *Rotation.as_euler(self.rotation, 'zxy'))
    
    def get_heading(self) -> float:
        """Gets the heading of the pose

        Returns:
            float: heading of the pose
        """
        heading = Rotation.as_euler(self.rotation, 'xyz')[1]
        return heading
    
    def get_heading_degrees(self) -> float:
        """Gets the heading of the pose in degrees

        Returns:
            float: heading of the pose in degrees
        """
        return self.get_heading()*180/np.pi
    
    def get_x(self) -> float:
        """Gets the x component of the pose translation

        Returns:
            float: x component of the pose translation
        """
        return self.x
    
    def get_x_inches(self) -> float:
        """Gets the x component of the pose translation in inches

        Returns:
            float: x component of the pose translation in inches
        """
        return self.x*39.3701
    
    def get_y(self) -> float:
        """Gets the y component of the pose translation

        Returns:
            float: y component of the pose translation
        """
        return self.y
    
    def get_y_inches(self) -> float:
        """Gets the y component of the pose translation in inches

        Returns:
            float: y component of the pose translation in inches
        """
        return self.y*39.3701

    def get_z(self) -> float:
        """Gets the z component of the pose translation

        Returns:
            float: z component of the pose translation
        """
        return self.z
    
    def get_z_inches(self) -> float:
        """Gets the z component of the pose translation in inches

        Returns:
            float: z component of the pose translation in inches
        """
        return self.z*39.3701
    
    def get_roll(self) -> float:
        """Gets the roll of the pose

        Returns:
            float: roll of the pose
        """
        return Rotation.as_euler(self.rotation, 'xyz')[0]

    def get_pitch(self) -> float:
        """Gets the pitch of the pose

        Returns:
            float: pitch of the pose
        """
        return Rotation.as_euler(self.rotation, 'xyz')[1]
    
    def get_yaw(self) -> float:
        """Gets the yaw of the pose

        Returns:
            float: yaw of the pose
        """
        return Rotation.as_euler(self.rotation, 'xyz')[2]
    
    def get_translation(self) -> npt.NDArray[np.float32]:
        """Gets the translation of the pose

        Returns:
            npt.NDArray[np.float32]: (3,) translation vector
        """
        return np.array([self.x, self.y, self.z])
    
    def get_translation_inches(self) -> npt.NDArray[np.float32]:
        """Gets the translation of the pose in inches

        Returns:
            npt.NDArray[np.float32]: (3,) translation vector in inches
        """
        return self.get_translation() * 39.3701
    
    def get_orientation(self) -> npt.NDArray[np.float32]:
        """Gets the orientation of the pose

        Returns:
            npt.NDArray[np.float32]: (3,) orientation vector
        """
        return Rotation.as_euler(self.rotation, 'xyz')
    
    def get_depth(self) -> np.float32:
        """Gets the depth in the xz plane

        Returns:
            np.float32: depth in the xz plane
        """
        return np.linalg.norm([self.x, self.z])
    
    def get_depth_inches(self) -> np.float32:
        """Gets the depth in the xz plane in inches

        Returns:
            np.float32: depth in the xz plane in inches
        """
        return np.linalg.norm([self.x, self.z])*39.3701
    
    def get_transformation_matrix(self) -> npt.NDArray[np.float32]:
        """Gets the 4x4 transformation matrix of the pose

        Returns:
            npt.NDArray[np.float32]: (4, 4) transformation matrix
        """
        # print("rotation", Rotation.as_euler(self.rotation, 'xyz'))
        return np.block([[self.rotation.as_matrix(), self.get_translation().reshape(3, 1)], [0, 0, 0, 1]])
    
    @staticmethod
    def from_transformation_matrix(matrix: npt.NDArray[np.float32]) -> Pose:
        """Creates a Pose object from a 4x4 transformation matrix

        Args:
            matrix (npt.NDArray[np.float32]): (4, 4) transformation matrix

        Returns:
            Pose: Pose object
        """
        return Pose(*matrix[0:3, 3], *Rotation.from_matrix(matrix[0:3, 0:3]).as_euler('xyz', degrees=False))
    
    # Can be used to "average" poses
    def __add__(self, other: Pose) -> Pose:
        """Adds two poses together element-wise (both translation and rotation)

        Args:
            other (Pose): other pose to add

        Returns:
            Pose: sum of the two poses
        """
        r_sum = Rotation.from_euler('xyz', Rotation.as_euler(self.rotation, 'xyz') + Rotation.as_euler(other.rotation, 'xyz'))
        t_sum = (self.get_translation() + other.get_translation()) 
        return Pose(*t_sum, *r_sum.as_euler('xyz', degrees=False))
    
    def __mult__(self, other: float) -> Pose:
        """Multiplies each element of the pose by a scalar (both translation and rotation)

        Args:
            other (float): scalar to multiply by

        Returns:
            Pose: scaled pose
        """
        return Pose(*(self.get_translation() * other), *(Rotation.as_euler(self.rotation, 'xyz') * other))
        
