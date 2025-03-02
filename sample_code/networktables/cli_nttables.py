import random
import time

from networktables import NetworkTables

# Replace" localhost" with the robot’s IP if remote
# IP = "10.32.5.2"
IP = "localhost"

# Create a NetworkTables instance
# Connect to the server (Replace with correct IP if needed)
NetworkTables.initialize("192.168.1.19")

# Get the tag table
sd = NetworkTables.getTable("tags")

# send data to server
print("Client started, waiting for data...")


def make_tags():
    n = random.randint(1, 5)
    tags = []

    for i in range(n):
        id = random.randint(1, 22)
        angle = random.random() * 360 - 180
        d = random.random() * 10
        tags.extend([i, angle, d])
    return tags


i = 0
try:
    while True:
        print(i)
        # send values
        data = make_tags()
        data.insert(0, i)
        print(data)
        sd.putNumberArray("tags", data)

        i += 1
        time.sleep(1)  # update every second
except KeyboardInterrupt:
    print("Shutting down client.")
