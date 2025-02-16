import ntcore


class NetworkTable:

    def __init__(self, cfg, table):
        # Create a NetworkTables instance
        self.ntinst = ntcore.NetworkTableInstance.getDefault()
        self.ntinst.startClient4("jetson")  # Start client with name "jetson"
        # condnect to the server
        self.ntinst.setServer(cfg["robot"]["ip"])
        # Get the tag table
        self.table = self.ntinst.getTable(table)

    def send_array(self, name, data):
        self.table.putNumberArray(name, data)

    def send(self, name, data):
        self.table.putNumber(name, data)
