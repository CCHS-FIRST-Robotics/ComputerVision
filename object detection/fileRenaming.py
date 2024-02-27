import os
filepath = "C:\\GitHub\\2024ComputerVision\\object detection\\Training-Images\\ffmpeg-2024-02-04-git-7375a6ca7b-full_build\\batch-"
final_filepath = "C:\\GitHub\\2024ComputerVision\\object detection\\Training-Images\\ffmpeg-2024-02-04-git-7375a6ca7b-full_build\\ffmpeg-2024-02-04-git-7375a6ca7b-full_build\\bin\\images"
files = []
counter = 1057 # incase we add more images this is already all set
for i in range(1, 9):
    files = os.listdir(filepath + str(i))
    for f in files:
            os.rename(filepath + str(i) + "\\" + f, final_filepath + "\\note-" + str(counter) + ".png")
            counter += 1
            
# we have 1056 images! <3 #