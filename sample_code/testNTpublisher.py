import ntcore
import time

class EasyNetworkTableExample():
    def __init__(self) -> None:

        inst = ntcore.NetworkTableInstance.getDefault()

        table = inst.getTable("datatable")

        self.xPub = table.getDoubleTopic("x").publish()
        self.yPub = table.getDoubleTopic("y").publish()
        self.x = 0
        self.y = 0
    def teleopPeriodic(self) -> None:
        self.xPub.set(self.x)
        self.yPub.set(self.y)
        self.x += 0.05
        self.y += 1.0

tableData = EasyNetworkTableExample()

i = 0
while True:
    tableData.teleopPeriodic()

    time.sleep(1)
    print("publish", i)
    i+=1