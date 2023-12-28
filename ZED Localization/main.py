# Local Imports
import time
from zed_detector import ZEDDetector
from april_tag import AprilTag
from pose import Pose

# Global Imports
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation #type: ignore
import cv2 
import pyzed.sl as sl # type: ignore
import ntcore


# Initialize networktables instance
n_table = ntcore.NetworkTableInstance.getDefault()
# n_table.setServerTEAM(3205) # Connects to RIO server (running on robot)
n_table.setServer("10.0.0.81") # LAPTOP IPv4 ADDRESS (running on laptop/simulating robot code)
n_table.startClient4("Jetson Orin Nano (NT4)") # Any name will work

tags_table = n_table.getTable("tags")

tagPose2dPub = tags_table.getDoubleArrayTopic("pose_estimate").publish();
tagPose3dPub = tags_table.getDoubleArrayTopic("pose_estimate_3d").publish();

zedPosePub = tags_table.getDoubleArrayTopic("zed_pose_estimate").publish();
zedCovarPub = tags_table.getDoubleArrayTopic("zed_pose_covar").publish();

primaryTagIdPub = tags_table.getDoubleTopic("primary_tag_id").publish();
primaryTagXPub = tags_table.getDoubleTopic("primary_tag_x").publish();
primaryTagYPub = tags_table.getDoubleTopic("primary_tag_y").publish();
primaryTagZPub = tags_table.getDoubleTopic("primary_tag_z").publish();
primaryTagRollPub = tags_table.getDoubleTopic("primary_tag_roll").publish();
primaryTagPitchPub = tags_table.getDoubleTopic("primary_tag_pitch").publish();
primaryTagHeadingPub = tags_table.getDoubleTopic("primary_tag_heading").publish();

tagIdsPub = tags_table.getDoubleArrayTopic("tag_ids").publish();
tagXsPub = tags_table.getDoubleArrayTopic("tag_xs").publish();
tagYsPub = tags_table.getDoubleArrayTopic("tag_ys").publish();
tagZsPub = tags_table.getDoubleArrayTopic("tag_zs").publish();
tagRollsPub = tags_table.getDoubleArrayTopic("tag_rolls").publish();
tagPitchesPub = tags_table.getDoubleArrayTopic("tag_pitches").publish();
tagHeadingsPub = tags_table.getDoubleArrayTopic("tag_headings").publish();

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
cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Image", *detector.image_size)
print(*detector.image_size) # NOTE THIS IS ONLY LEFT CAM (only one used atm)

# Set primary method of pose estimation (what's sent over NT)
primary = "pnp_pose"
fps_lst = []
while True:
    start = time.time()
    # Run the periodic function to update the image, depth, and pose data
    # Returns True if a new image is available, False otherwise
    if not detector.periodic():
        continue
    timestamp = detector.timestamp
    
    pose = None
    match primary:
        case "zed_pose":
            pose = detector.get_camera_pose_zed()
        case "pnp_pose":
            pose = detector.get_camera_pose_pnp()
        case "depth_pose":
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
    for tag in detector.get_detected_tags():
        print(f"Tag: {tag.id}")
    print(f"{pose} at {timestamp}ms")
    
    pose_2d = pose.get_2d_pose().tolist()
    # Convert from ZED (x, z, pitch) to WPILib (x, y, yaw)
    pose_2d = [pose_2d[1], pose_2d[0], pose_2d[2]]
    tagPose2dPub.set(pose_2d)
    
    pose_3d = pose.get_3d_pose().tolist()
    # Convert from ZED (x, y, z, roll, pitch, yaw) to WPILib (z, x, y, yaw, roll, pitch)
    pose_3d = [pose_3d[2], pose_3d[0], pose_3d[1], pose_3d[5], pose_3d[3], pose_3d[4]]
    tagPose3dPub.set(pose_3d)
    
    tags = detector.get_detected_tags()
    tags_and_poses = []
    tag_ids = []
    tag_xs = []
    tag_ys = []
    tag_zs = []
    tag_rolls = []
    tag_pitches = []
    tag_headings = []
    
    for tag in tags:
        tag_pose = detector.get_tag_pose(tag)
        if not tag_pose:
            continue
        
        tag_ids.append(float(tag.id))

        wpi_pose = tag_pose.get_as_wpi_pose()
        tag_xs.append(wpi_pose.get_x())
        tag_ys.append(wpi_pose.get_y())
        tag_zs.append(wpi_pose.get_z())
        tag_rolls.append(wpi_pose.get_roll())
        tag_pitches.append(wpi_pose.get_pitch())
        tag_headings.append(wpi_pose.get_yaw())
        
        tags_and_poses.append((tag, tag_pose))
    
    tagIdsPub.set(tag_ids)
    tagXsPub.set(tag_xs)
    tagYsPub.set(tag_ys)
    tagZsPub.set(tag_zs)
    tagRollsPub.set(tag_rolls)
    tagPitchesPub.set(tag_pitches)
    tagHeadingsPub.set(tag_headings)
    
    tags_and_poses.sort(key=lambda tag: tag[1].get_depth())
    primary_tag = tags_and_poses[0] if tags_and_poses else None
    if primary_tag:
        wpi_pose = primary_tag[1].get_as_wpi_pose()

        primaryTagIdPub.set(primary_tag[0].id)
        primaryTagXPub.set(wpi_pose.get_x())
        primaryTagYPub.set(wpi_pose.get_y())
        primaryTagZPub.set(wpi_pose.get_z())
        primaryTagRollPub.set(wpi_pose.get_roll())
        primaryTagPitchPub.set(wpi_pose.get_pitch())
        primaryTagHeadingPub.set(wpi_pose.get_yaw())
    else:
        primaryTagIdPub.set(-1)
        primaryTagXPub.set(-1)
        primaryTagYPub.set(-1)
        primaryTagZPub.set(-1)
        primaryTagRollPub.set(-1)
        primaryTagPitchPub.set(-1)
        primaryTagHeadingPub.set(-1)
    
    fps = 1/(float(time.time() - start))
    if len(fps_lst) < 100:
        fps_lst.append(fps)
    else:
        fps_lst.pop(0)
        fps_lst.append(fps)

    print(f"FPS: {round(fps)}, avg: {sum(fps_lst)/len(fps_lst)}")
    image = detector.get_image()
    cv2.imshow("Image", image)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
