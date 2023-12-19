# Local Imports
from zed_detector import ZEDDetector
from april_tag import AprilTag
from pose import Pose

# Global Imports
import numpy as np
##import numpy.typing as npt
from scipy.spatial.transform import Rotation #type: ignore
import cv2 
import pyzed.sl as sl # type: ignore
from networktables import NetworkTables
import time


# Initialize networktables instance
# n_table = ntcore.NetworkTableInstance.getDefault()
# # n_table.setServerTEAM(3205) # Connects to RIO server (running on robot)
# n_table.setServer("10.0.0.81") # LAPTOP IPv4 ADDRESS (running on laptop/simulating robot code)
# n_table.startClient4("Jetson NT") # Any name will work

n_table = NetworkTables.initialize(server="10.32.5.2")
tags_table = NetworkTables.getTable("tags")

# pose2dPub = tags_table.getDoubleArrayTopic("pose_estimate").publish();
# pose3dPub = tags_table.getDoubleArrayTopic("pose_estimate_3d").publish();

# primaryTagIdPub = tags_table.getDoubleTopic("primary_tag_id").publish();
# primaryTagXPub = tags_table.getDoubleTopic("primary_tag_x").publish();
# primaryTagYPub = tags_table.getDoubleTopic("primary_tag_y").publish();
# primaryTagZPub = tags_table.getDoubleTopic("primary_tag_z").publish();
# primaryTagHeadingPub = tags_table.getDoubleTopic("primary_tag_heading").publish();

# tagIdsPub = tags_table.getDoubleArrayTopic("tag_ids").publish();
# tagXsPub = tags_table.getDoubleArrayTopic("tag_xs").publish();
# tagYsPub = tags_table.getDoubleArrayTopic("tag_ys").publish();
# tagZsPub = tags_table.getDoubleArrayTopic("tag_zs").publish();
# tagHeadingsPub = tags_table.getDoubleArrayTopic("tag_headings").publish();


# Create a ZED camera
zed = sl.Camera()

# Create configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_minimum_distance = .3

# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()

# Setting the depth confidence parameters
# runtime_parameters.enable_fill_mode = True
runtime_parameters.confidence_threshold = 90

tracking_parameters = sl.PositionalTrackingParameters()

detector = ZEDDetector(zed, init_params, runtime_parameters, tracking_parameters, 0.1524, Pose(0, 0, 0, 0, 0, 0))


# Resize the window to match the camera resolution (VERY IMPORTANT FOR TESTING -- WONT SHOW FULL IMAGE OTHERWISE)
## cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 
## cv2.resizeWindow("Image", *detector.image_size)
## print(*detector.image_size) # NOTE THIS IS ONLY LEFT CAM (only one used atm)

# Set primary method of pose estimation (what's sent over NT)
primary = "pnp_pose"
while True:
    start = time.time()    
    # Run the periodic function to update the image, depth, and pose data
    # Returns True if a new image is available, False otherwise
    if not detector.periodic():
        continue
    timestamp = detector.timestamp
    
    pose = None
    if primary == "zed_pose":
        pose = detector.get_camera_pose_zed()
    elif primary == "pnp_pose":
        pose = detector.get_camera_pose_pnp()
    elif primary == "depth_pose":
        pose = detector.get_camera_pose_depth_average()
    
    if pose:
        # Tranform the pose from the camera frame to the robot frame
        # pose = ZEDDetector.get_robot_pose(pose)
        pass
    else:
        # If no pose is available, set the pose to a default value
        pose = Pose(-1, -1, -1, -1, -1, -1)

    # For testing:
    print()
    print(f"Tag: {tag.id}")
    print(f"{pose} at {timestamp}ms")
    
    pose_2d = pose.get_2d_pose().tolist()
    #Convert from ZED (x, z, pitch) to WPILib (x, y, yaw)
    pose_2d = [pose_2d[1], pose_2d[0], pose_2d[2]]
    tags_table.putNumberArray("pose_estimate", pose_2d)
    
    pose_3d = pose.get_3d_pose().tolist()
    # Convert from ZED (x, y, z, roll, pitch, yaw) to WPILib (x, y, z, roll, pitch, yaw)
    pose_3d = [pose_3d[2], pose_3d[0], pose_3d[1], pose_3d[5], pose_3d[3], pose_3d[4]]
    tags_table.putNumberArray("pose_estimate_3d", pose_3d)
    
    tags = detector.get_detected_tags()
    tags_and_poses = []
    tag_ids = []
    tag_ys = []
    tag_xs = []
    tag_headings = []
    
    #for tag in tags:
    #    tag_pose = detector.get_tag_pose(tag)
    #    if not tag_pose:
    #        continue
    #    
    #    tag_ids.append(float(tag.id))
    #    tag_xs.append(tag_pose.get_z())
    #    tag_ys.append(tag_pose.get_x())
    #    tag_headings.append(tag_pose.get_heading())
    #    
    #    tags_and_poses.append((tag, tag_pose))
    
    #tags_table.putNumberArray("tag_ids", tag_ids)
    #tags_table.putNumberArray("tag_xs", tag_xs)
    #tags_table.putNumberArray("tag_ys", tag_ys)
    # tags_table.putNumberArray("tag_zs", tag_zs)
   # tags_table.putNumberArray("tag_headings", tag_headings)
   # 
    #tags_and_poses.sort(key=lambda tag: tag[1].get_depth())
    #primary_tag = tags_and_poses[0] if tags_and_poses else None
    #if primary_tag:
    #    tags_table.putNumber("primary_tag_id", primary_tag[0].id)
    #    tags_table.putNumber("primary_tag_x", primary_tag[1].get_z())
    #    tags_table.putNumber("primary_tag_y", primary_tag[1].get_x())
    #    # tags_table.putNumber("primary_tag_z", primary_tag[1].get_y())
    #    tags_table.putNumber("primary_tag_heading", primary_tag[1].get_heading())
    #else:
    #    tags_table.putNumber("primary_tag_id", -1)
    #    tags_table.putNumber("primary_tag_x", -1)
    #    tags_table.putNumber("primary_tag_y", -1)
    #    # tags_table.putNumber("primary_tag_z", -1)
    #    tags_table.putNumber("primary_tag_heading", -1)
    
    print(f"FPS: {1/(float(time.time() - start))}")
    ## image = detector.get_image()
    ## cv2.imshow("Image", image)
    
    ## key = cv2.waitKey(1)
    ## if key == ord('q'):
    # #    break
    
