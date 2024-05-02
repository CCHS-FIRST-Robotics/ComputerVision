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
import ntcore

# Local imports
from pose import Pose
from note_detector import ZEDDetector

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

    # create a networktable
    n_table = ntcore.NetworkTableInstance.getDefault()
    n_table.setServerTeam(3205) # Connects to RIO server (running on robot)
    # n_table.setServer("10.0.0.81") # LAPTOP IPv4 ADDRESS (running on laptop/simulating robot code)
    n_table.startClient4("Jetson Orin Nano (NT4)") # Any name will work

    noteDisplacementPub = n_table.getDoubleArrayTopic('note_displacements').publish()
    
    tracking_parameters = sl.PositionalTrackingParameters()
    
    detector = ZEDDetector(zed, init_params, runtime_parameters, tracking_parameters, Pose(0,0,0,0,0,0))
    
    while True:
        
        # Run the periodic function to update the image, depth, and pose data
        # Returns True if a new image is available, False otherwise
        if not detector.periodic():
            continue
        
        
        pose = None
        notes = detector.get_notes()
        note_displacements = []
        # loop through the notes and put the displacements in a list
        for note in notes:
            displacement = detector.get_object_pose(note)
            note_displacements.append(displacement)

        # publish to note_displacements
        noteDisplacementPub.set(note_displacements)
                
        if pose:
            print(f'Object pose estimated at: {pose}')
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break