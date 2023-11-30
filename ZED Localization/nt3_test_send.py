from networktables import NetworkTables


# Initialize networktables instance
n_table = NetworkTables.initialize(server="10.0.0.81")

tags_table = NetworkTables.getTable("tags")

while True:
    tags_table.putNumber("primary_tag_x", 5.0)
