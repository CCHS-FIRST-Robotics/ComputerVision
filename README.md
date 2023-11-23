# 2024 CCHS Computer Vision

Project inlcudes 
- April tag detections using opencv aruco library
- yolo object detection
- video streaming

## Installation
### Setup
Ensure you're running the correct version of python with ```python --version``` (or ```python3 --version```)

```python -m pip install -r requirements.txt```

Install the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
Install the [ZED Python API](https://www.stereolabs.com/docs/app-development/python/install/)

## April Tags
Apriltag Marker Detection with [OpenCV ArUco](https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html)

## Yolo
You Only Look Once

YOLO should (generally) be ran on the GPU. If your computer has an NVIDIA GPU, make sure you install CUDA and test that it is working properly [testing instructions here](https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu)

## Video Streaming
[Video streaming code](videostreaming/README.md)

## Tutorials and HOWTOs
[Computer vision tutorials](docs/README.md)

## Guides
- [Python environment setup](docs/pyenv_setup.md)
- [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/)
- [Adding your work](docs/adding_your_work.md)
