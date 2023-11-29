import ntcore


# Initialize networktables instance
n_table = ntcore.NetworkTableInstance.getDefault()

# Either start or connect to server # 
n_table.startServer()

# Either start or connect to server # 
# n_table.setServer("10.0.0.81") # LAPTOP IPv4 ADDRESS
# n_table.startClient4("Jetson NT") # Any name will work


tags_table = n_table.getTable("tags")

num = tags_table.getDoubleTopic("num").publish()

while True:
    num.set(1.0)