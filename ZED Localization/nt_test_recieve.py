import ntcore
import time
import logging
logging.basicConfig(level=logging.DEBUG)


# Initialize networktables instance
n_table = ntcore.NetworkTableInstance.getDefault()

# Either start or connect to server # 
n_table.startServer() 

# Either start or connect to server # 
# n_table.setServer("10.0.0.81") # LAPTOP IPv4 ADDRESS
# n_table.startClient4("Jetson NT") # Any name will work

tags_table = n_table.getTable("tags")

num = tags_table.getDoubleTopic("num").subscribe(0)

i = 0
while True:
    i += 1
    print(f"loop {i}: {num.get()}")
    time.sleep(1)