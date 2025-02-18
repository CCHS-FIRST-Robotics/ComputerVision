from networktables import NetworkTables


class NetworkTable:

    def __init__(self, cfg, table):
        # condnect to the server
        NetworkTables.initialize(cfg["robot"]["ip"])
        # Get the tag table
        self.table = NetworkTables.getTable(table)

    def send_array(self, name, data):
        self.table.putNumberArray(name, data)

    def send(self, name, data):
        self.table.putNumber(name, data)
