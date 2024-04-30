# Global imports
import pyzed.sl as sl #type: ignore 
import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation #type: ignore
import sys
import math
from typing import List, Tuple, Union
from ultralytics import YOLO

# Local imports
from pose import Pose

class ZEDDetector:
    
    # Transformation matrix from the camera frame to the robot frame
    # camera_to_robot_transformation = Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).get_transformation_matrix()
    model = YOLO("object detection\\note_detection_weights.pt")
    
    def __init__(self, zed, init_params, runtime_params, tracking_params, initial_pose: Pose) -> None:
        """Initializes the ZED camera

        Args:
            zed (sl.Camera): ZED Camera object
            init_params (sl.InitParameters): Init parameters for the ZED camera
            runtime_params (sl.RuntimeParameters): Runtime parameters for the ZED camera
            tracking_params (sl.PositionalTrackingParameters): Positional Tracking parameters for the ZED camera
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
        
        
        # Initialize the variables updated in the periodic function
        self.annotated_image = np.array([])
        self.timestamp = 0
        self.notes = []
        
        # Set the initial pose of the camera
        self.intial_pose = initial_pose
    
    def periodic(self) -> bool:
        """Update the depth, image, and pose data from the ZED

        Returns:
            bool: True if a new image is available, False otherwise
        """
        # if no new image, stop
        err = self.zed.grab(self.runtime_params)
        if err != sl.ERROR_CODE.SUCCESS: # A new image is available if grab() returns SUCCESS
            return False
        
        # find camera pose
        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT) # Get the left image
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH) # Retrieve depth Mat. Depth is aligned on the left image
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA) # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.get_position(self.zed_pose, sl.REFERENCE_FRAME.WORLD) # Retrieve camera pose
        # self.timestamp = self.zed_pose.timestamp
        self.timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        
        image = cv2.cvtColor(self.image_zed.get_data(), cv2.COLOR_BGRA2BGR)
        image_debug = self.image_zed.get_data()
        self.annotated_image = image_debug
        
        # get a list of all notes detected
        self.notes = []
        results = self.model(image, stream=True)
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                print(conf)
                if cls == 0 and conf > 0.8:
                    self.notes.append(b)
                    
        print(len(self.notes))
        return True
    
    def get_image(self) -> cvt.MatLike:
        """Gets the latest image from the ZED camera

        Returns:
            cvt.MatLike: Latest image
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

        print("Distance to Camera at ({}, {}) (image center): {:1.3} m.".format(x, y, np.linalg.norm(point_cloud_value[0:3])), end="\r")
        # sys.stdout.flush()
        # print(point_cloud_value)
        return np.array(point_cloud_value[0:3])
    
    def get_object_pose(self, note) -> npt.NDArray[np.float32] | None:
        # get detected objects in the image
        # results = self.model(image, stream=True)
        # for r in results:
                # get what it thinks it is & check if it is #44 (note)
        x1, y1, x2, y2 = note.xyxy[0]
        x1, y1, x2, y2, = int(x1), int(y1), int(x2), int(y2)
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        coords = x_center, y_center
        
           
        displacement = self.get_displacement(coords)
        if not np.isnan(displacement[0]):
            return displacement
        
    def draw_object(self, note, image):
        """Draws a box on the image outlining the detected object. Also shows the object's 
        distance from the camera and its confidence.

        Args:
            note (_type_): The detected object
            image (_type_): The image that contains the object
        """
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        x1, y1, x2, y2 = note.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        conf = note.conf[0]
        displacement = self.get_object_pose(note)
        d_txt = f"{np.round(np.linalg.norm(displacement), 2)}" if displacement is not None else "NaN" # type: ignore
        txt = f"Note {conf:.2f} at {d_txt}m" 

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(image, txt, (x1, y1), font, fontScale, color, thickness)
        
    def get_notes(self):
        return self.notes
    
    def get_camera_pose_zed(self) -> Pose:
        """Get the camera pose using the ZED Visual Odometry

        Returns:
            Pose: Estimated camera pose in the world frame
        """
        # Reformat orientation from YZX to XYZ
        orientation = self.zed_pose.get_euler_angles(radian=True)
        orientation = [orientation[2], orientation[0], orientation[1]]
        return self.intial_pose + Pose(*self.zed_pose.get_translation().get(), *orientation)

if __name__ == '__main__':
    
    # Create a ZED camera
    zed = sl.Camera()
    
    # Create configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720 # yolo code only works with 720
    init_params.depth_minimum_distance = .3
    
    # Create set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode (I think this is deprecated now??)
    
    # Setting the depth confidence parameters
    # runtime_parameters.enable_fill_mode = True
    runtime_parameters.confidence_threshold = 90
    #runtime_parameters.textureness_confidence_threshold = 90 (deprecated???)
    
    tracking_parameters = sl.PositionalTrackingParameters()
    
    detector = ZEDDetector(zed, init_params, runtime_parameters, tracking_parameters, Pose(0,0,0,0,0,0))
    
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
        
        pose = None
        image = detector.get_image()
        
        notes = detector.get_notes()
        for note in notes:
            displacement = detector.get_object_pose(note)
            detector.draw_object(note, image)
                
        if pose:
            print(f'Object pose estimated at: {pose}')
        
        cv2.imshow("Image", image)
        
        key = cv2.waitKey(1)
        if key == ord('r'):
            recording = not recording
        
        if key == ord('q'):
            break
        
