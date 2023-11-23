# Global imports
import pyzed.sl as sl #type: ignore 
import cv2
from cv2 import aruco # NOTE: This is the opencv-contrib-python package, not the opencv-python package
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation #type: ignore
import sys
import math
from typing import List, Tuple, Union

# Local imports
from pose import Pose
from april_tag import AprilTag

class ZEDDetector:
    
    # Transformation matrix from the camera frame to the robot frame
    camera_to_robot_transformation = Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).get_transformation_matrix()
    
    def __init__(self, zed, init_params, runtime_params, tracking_params, tag_size: float, initial_pose: Pose) -> None:
        """Initializes the ZED camera and AprilTag detector

        Args:
            zed (sl.Camera): ZED Camera object
            init_params (sl.InitParameters): Init parameters for the ZED camera
            runtime_params (sl.RuntimeParameters): Runtime parameters for the ZED camera
            tracking_params (sl.PositionalTrackingParameters): Positional Tracking parameters for the ZED camera
            tag_size (float): Size (length) of the AprilTag in meters
            initial_pose (Pose): Initial pose of the camera in the world frame
        """
        self.zed = zed
        err = self.zed.open(init_params)
        
        if (err!=sl.ERROR_CODE.SUCCESS):
            print(err)
            exit(-1)
        
        self.runtime_params = runtime_params
        
        self.tracking_params = tracking_params
        err = zed.enable_positional_tracking(tracking_params)
        if (err!=sl.ERROR_CODE.SUCCESS):
            print(err)
            exit(-1)
        
        # Initialize image, depth, and point cloud matrices
        self.image_zed = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        self.zed_pose = sl.Pose()
        
        # Get camera information (image size, camera intrinsics)
        self.resolution = self.zed.get_camera_information().camera_configuration.resolution
        self.image_size = [self.resolution.width, self.resolution.height]
        
        self.calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
        self.fovh = self.calibration_params.left_cam.h_fov
        self.fovv = self.calibration_params.left_cam.v_fov
        
        self.fx = self.calibration_params.left_cam.fx
        self.fy = self.calibration_params.left_cam.fy
        self.cx = self.calibration_params.left_cam.cx
        self.cy = self.calibration_params.left_cam.cy
        
        self.left_camera_matrix = np.array([
            [self.fx, 0, self.cx], 
            [0, self.fy, self.cy], 
            [0,       0,       1]
        ])
        self.left_distortion = np.array(self.calibration_params.left_cam.disto)
        
        # Initialize the aruco dictionary and parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16H5)
        self.parameters = aruco.DetectorParameters()
        self.tag_size = tag_size
        
        # Initialize the variables updated in the periodic function
        self.annotated_image = np.array([])
        self.timestamp = 0
        self.detected_tags: List[AprilTag] = []
        
        # Set the initial pose of the camera
        self.intial_pose = initial_pose
    
    def periodic(self) -> bool:
        """Update the depth, image, and pose data from the ZED, and update the detected tags list

        Returns:
            bool: True if a new image is available, False otherwise
        """
        err = self.zed.grab(self.runtime_params)
        if err != sl.ERROR_CODE.SUCCESS: # A new image is available if grab() returns SUCCESS
            return False
        
        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT) # Get the left image
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH) # Retrieve depth Mat. Depth is aligned on the left image
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA) # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.get_position(self.zed_pose, sl.REFERENCE_FRAME.WORLD) # Retrieve camera pose
        self.timestamp = self.zed_pose.timestamp
        
        image = cv2.cvtColor(self.image_zed.get_data(), cv2.COLOR_BGR2GRAY)
        image_debug = self.image_zed.get_data()
        
        self.detected_tags = []
        tag_corners, tag_ids, rejected = aruco.detectMarkers(image, self.aruco_dict, None, parameters=self.parameters)
        for i, corners in enumerate(tag_corners):
            tag = AprilTag(tag_ids[i][0], self.tag_size, "16h5", corners[0])
            self.detected_tags.append(tag)
            tag.draw_tag(image_debug)
        self.annotated_image = image_debug
        
        return True
    
    def get_detected_tags(self) -> List[AprilTag]:
        """Gets the list of detected AprilTags

        Returns:
            List[AprilTag]: List of detected AprilTags
        """
        return self.detected_tags
    
    def get_image(self) -> cvt.MatLike:
        """Gets the latest image from the ZED camera (annotated with detected AprilTags)

        Returns:
            cvt.MatLike: Latest image (annotated with detected AprilTags)
        """
        return self.annotated_image
    
    def get_displacement(self, point: Tuple[int, int]) -> npt.NDArray[np.float32]:
        """Gets the displacement vector from the ZED point cloud at the given pixel coordinates

        Args:
            point (Tuple[int, int]): Point of interest in pixel coordinates

        Raises:
            Exception: Exception during ZED point cloud access (I forget what this is for)

        Returns:
            npt.NDArray[np.float32]: (3,) displacement vector from point cloud
        """
        x, y = point
    
        try:
            err, point_cloud_value = self.point_cloud.get_value(x, y)
            if (err!=sl.ERROR_CODE.SUCCESS):
                print(err)
        except Exception as e:
            print("BAD THING HAPPENED")
            print(e)
            print(x, y)
            return np.array([-1, -1, -1])
            raise Exception

        #print("Distance to Camera at ({}, {}) (image center): {:1.3} m.".format(x, y, distance), end="\r")
        # sys.stdout.flush()
        # print(point_cloud_value)
        return np.array(point_cloud_value[0:3])
    
    def get_tag_pose(self, tag: AprilTag) -> Pose | None:
        """Gets the pose of the given AprilTag relative to the robot

        Args:
            tag (AprilTag): AprilTag of interest

        Returns:
            Pose | None: Pose of the AprilTag relative to the robot; None if solvePnP fails
        """
        
        object_points = np.array(tag.get_corner_translations())
        image_points = np.array(tag.corners)
        
        retval, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.left_camera_matrix,
            self.left_distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        displacement = self.get_displacement(tag.center)
        if not np.isnan(displacement[0]):
            tvec = displacement
        
        rotation = Rotation.from_rotvec(rvec.T[0]).as_euler('xyz', degrees=False)
        
        if not retval:
            return None
        return self.get_robot_pose(Pose(*tvec, *rotation))
    
    def get_camera_pose_pnp(self) -> Pose | None:
        """Estimates the camera pose using SQPnP (https://www.ecva.net///papers/eccv_2020/papers_ECCV/papers/123460460.pdf)
        
        Object points are the corner poses (translation) of the detected tags in the world frame (known from field)
        
        Image points are the pixel coordinates of the detected tags' corners in the image frame

        Returns:
            Pose | None: Estimated camera pose in the world frame; None if the solvePnP function fails
        """
        if not self.detected_tags:
            return None
        
        object_points = np.array([tag.get_corner_translations() for tag in self.detected_tags])
        image_points = np.array([tag.corners for tag in self.detected_tags])
        
        # Find the camera pose using PnP
        # retval is a boolean indicating whether the function succeeded
        # rvec is a (3,) rotation vector, where the direction defines the plane of rotation and the magnitude defines the angle of rotation in radians
        # tvec is a (3,) displacement vector 
        retval, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.left_camera_matrix,
            self.left_distortion,
            flags=cv2.SOLVEPNP_SQPNP
        )

        # r_mat, jacobian = cv2.Rodrigues(rvec)
        
        # transform from rotation vector to euler angles (rvec is a (3, 1) matrix, refSormat to (3,) matrix)
        r = Rotation.from_rotvec(rvec.T[0])
        r = r * (Rotation.from_euler('xyz', [180, 0, 180], degrees=True)) # for some reason the natural orientation of the tag is upside down
        
        # debug:
        # print(retval)
        # print(r.as_euler('xyz', degrees=True))
        # print(tvec)
        
        if not retval:
            return None
        return Pose(*tvec, *r.as_euler('xyz', degrees=False))
    
    def get_camera_pose_depth_average(self) -> Pose | None:
        """Estimates the camera pose using a mix of SQPnP and the ZED point cloud. 
        Orientation is estimated using PnP, and translation is estimated using the point cloud
        
        See get_camera_pose_pnp() for more information on the PnP part of this function
        
        If the point cloud is nan at a tag center, that tag is ignored.
        If all the point clouds are nan, the translation is also estimated using the PnP function

        Returns:
            Pose | None: Estimated camera pose in the world frame; None if the solvePnP function fails
        """
        object_points = np.array([tag.get_corner_translations() for tag in self.detected_tags])
        image_points = np.array([tag.corners for tag in self.detected_tags])
        
        retval, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.left_camera_matrix,
            self.left_distortion,
            flags=cv2.SOLVEPNP_SQPNP
        )
        
        r = Rotation.from_rotvec(rvec.T[0])
        r = r * (Rotation.from_euler('xyz', [180, 0, 180], degrees=True))
        pitch = r.as_euler('xyz', degrees=False)[1]
        
        # Rotate the camera displacement vector <x, z> by "pitch" (in terms of the robot it's yaw) degrees, leave y fixed
        # Then, negate the x since the camera left is world right to get translation from tag to camera in world frame
        # Then, add the pose from the tag to the camera pose
        rotation = Rotation.from_euler('xyz', [0, -pitch, 0], degrees=False).as_matrix()
        world_translations = [
            rotation.dot(self.get_displacement(tag.center)) * np.array([-1, 1, 1]) + tag.get_world_translation() 
            for tag in self.detected_tags 
            if not np.isnan(self.get_displacement(tag.center)[0]) # ZED point cloud can return nan values
        ]
        
        # Take the average of all the translations found from the ZED point cloud and use that for the estimated camera pose translation 
        if world_translations:
            tvec = np.mean(world_translations, axis=0)
        
        if not retval:
            return None
        return Pose(*tvec, *r.as_euler('xyz', degrees=False))
        
    
    def get_camera_pose_zed(self) -> Pose:
        """Get the camera pose using the ZED Visual Odometry

        Returns:
            Pose: Estimated camera pose in the world frame
        """
        # Reformat orientation from YZX to XYZ
        orientation = self.zed_pose.get_euler_angles(radian=True)
        orientation = [orientation[2], orientation[0], orientation[1]]
        return self.intial_pose + Pose(*self.zed_pose.get_translation().get(), *orientation)
    
    # NOTE: shouldn't be static in the future, but we haven't decided on which method to use for camera pose estimation yet
    @staticmethod
    def get_robot_pose(camera_pose: Pose) -> Pose:
        """Get the robot pose from the camera pose

        Args:
            camera_pose (Pose): Camera pose in the world frame

        Returns:
            Pose: Robot pose in the world frame
        """
        return Pose.from_transformation_matrix(camera_pose.get_transformation_matrix().dot(ZEDDetector.camera_to_robot_transformation))
    
    
if __name__ == '__main__':
    
    # Create a ZED camera
    zed = sl.Camera()
    
    # Create configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Set the depth mode to performance (fastest)
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.depth_minimum_distance = .3
    
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode (I think this is deprecated now??)
    
    # Setting the depth confidence parameters
    # runtime_parameters.enable_fill_mode = True
    runtime_parameters.confidence_threshold = 90
    #runtime_parameters.textureness_confidence_threshold = 90 (deprecated???)
    
    tracking_parameters = sl.PositionalTrackingParameters()
    
    detector = ZEDDetector(zed, init_params, runtime_parameters, tracking_parameters, 0.1524, Pose(0, 0, 0, 0, 0, 0))
    
    recording = False
    primary = "pnp_pose"
    files = ["zed_pose.csv", "pnp_pose.csv", "depth_pose.csv"]
    for file in files:
        with open(file, 'w') as f:
            pass # clear the file
    measurement_spacing = 6 # inches
    while True:
        
        # Run the periodic function to update the image, depth, and pose data
        # Returns True if a new image is available, False otherwise
        if not detector.periodic():
            continue
        
        if recording:
            with open("zed_pose.csv", 'a') as f:
                # always returns a value (I think lol), since it doesn't use tags
                depth = detector.get_camera_pose_zed().get_translation_inches()[2]
                f.write(f"{round(depth / measurement_spacing) * measurement_spacing}, {depth}\n")
            
            with open("pnp_pose.csv", 'a') as f:
                depth = detector.get_camera_pose_pnp()
                # depth will return None if no tags are found or PnP fails
                if not depth:
                    f.write(f"{-1}, {-1}\n")
                    continue
                depth = depth.get_translation_inches()[2]
                f.write(f"{round(depth / measurement_spacing) * measurement_spacing}, {depth}\n")

            with open("depth_pose.csv", 'a') as f:
                depth = detector.get_camera_pose_depth_average()
                # depth will return None if no tags are found or PnP fails
                if not depth:
                    f.write(f"{-1}, {-1}\n")
                    continue
                depth = depth.get_translation_inches()[2]
                f.write(f"{round(depth / measurement_spacing) * measurement_spacing}, {depth}\n")
        
        pose = None
        match primary:
            case "zed_pose":
                pose = detector.get_camera_pose_zed()
            case "pnp_pose":
                pose = detector.get_camera_pose_pnp()
            case "depth_pose":
                pose = detector.get_camera_pose_depth_average()
                
        if pose:
            print(f'Robot pose estimated at: {pose}')
        
        image = detector.get_image()
        cv2.imshow("Image", image)
        
        key = cv2.waitKey(1)
        if key == ord('r'):
            recording = not recording
        
        if key == ord('q'):
            break
        