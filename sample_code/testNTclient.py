import ntcore
import time

if __name__ == "__main__":
    inst = ntcore.NetworkTableInstance.getDefault()
    table =  inst.getTable("datatable")
    xSub = table.getDoubleTopic("x").subscribe(0)
    ySub = table.getDoubleTopic("y").subscribe(0)
    inst.startClient4("Jetson Client")
    inst.setServerTeam(3205)
    inst.startDSClient()

    while True:
        time.sleep(1)
        x = xSub.get()
        y = ySub.get()
        print(f"X: {x} Y: {y}")