import sys
import time
from networktables import NetworkTables
import random
robotIP = "10.32.5.2"

NetworkTables.initialize(server = robotIP)

sd = NetworkTables.getTable("AprilTag info")

while True:
    markerID = random.randint(1, 22)
    print(time.time())
    sd.putNumberArray("apriltag", [markerID, 1, 23])
    time.sleep(1)