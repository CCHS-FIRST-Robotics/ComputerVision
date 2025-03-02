import time
from networktables import NetworkTables

def gotTags(key, value, isNew):
    print(key, value, isNew)


def main(args):
    #NetworkTables.initialize("localhost")
    NetworkTables.initialize()
    
    sd = NetworkTables.getTable("tags")
    #sd.addEntryListener(gotTags)

    try:
        while True:
            
            tags = sd.getNumberArray("tags", None)
            if tags is None:
                print(".")
            else:
                print(tags)
            time.sleep(1)

    except KeyboardInterrupt:
        print("done")
        
    
if __name__ == "__main__":
    args = None
    main(args)

