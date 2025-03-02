# This is a NetworkTables server demo
import time
import numpy as np

import ntcore
from ntcore import NetworkTableInstance

TEAM = 3205

# Create a NetworkTables instance
ntinst = NetworkTableInstance.getDefault()
ntinst.setServerTeam(TEAM)
ntinst.startServer()  # Start server on default port 1735

# print default port
print("PORT", ntcore.NetworkTableInstance.kDefaultPort4)

# Create a table named "tags"
sd = ntinst.getTable("tags")
print("Server started...")

try:
    while True:
        # print table values
        # all tags are in flattened array
        # format [packetid, tagid1, angle1, distance1, tagid2, angle2, distance2]
        tags = sd.getNumberArray("tags", None)

        if tags is not None:
#            print(tags)
            pid = tags[0]
            n = (len(tags) - 1) // 4

            for i in range(n):
                ix = i * 4 + 1
                id = int(tags[ix])
                angleh = tags[ix+2] * 180 / np.pi
                anglev = tags[ix+3] * 180 / np.pi
                d = tags[ix+4]
                print(f"id {id} {angleh:.1f} {anglev:.1f} {d:.2f}")
#            print("packetid", tags[0])
#            n = (len(tags) - 1) // 3
#            print("count", n)
#            for i in range(n):
#                ix = i * 3 + 1
#                print(tags[ix : ix + 3])
        print(".")
        time.sleep(1)  # Update every second
except KeyboardInterrupt:
    print("Shutting down server.")
    NetworkTableInstance.destroy(ntinst)
