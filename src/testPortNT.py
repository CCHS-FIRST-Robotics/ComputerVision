import ntcore

inst = ntcore.NetworkTableInstance.getDefault()
port = inst.getPort()
print("NetworkTables port:", port)