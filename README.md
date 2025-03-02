# CCHS Computer Vision

Project is using jetson orin nano single board computer and arducam quad rolling shutter cameras.  

Processes running in parallel using shared memory IPC. This enables running processes running independently and at different frame rates with low latency image capture and processing.

Task currently supported
- April tag detections
- Yolo object detection
- Video streaming

## Installation
### Setup
```pip install -r requirements.txt```

## CCVision
Run locally  
```sh run.sh```

Start/stop service  
```sudo systemctl start ccvision.service```
```sudo systemctl stop ccvision.service```

### Tools

Image capture  
```python -m tools.capture```

Camera calibration  
```python tools/calibrate.py -d calibration_images``` 

Marker Generator  
```python tools/marker_gen.py```

Chessboard Generator  
```python tools/chessboard.py```

Auto bounding box detection  
```python tools/auto_bb.py -d imgdir```

### Unit Tests
python -m unittest

## April Tags
Aruco Marker Detection

## Yolo
[You Only Look Once Quickstart](https://github.com/ultralytics/ultralytics/blob/main/docs/en/quickstart.md)  
[Yolov Getting Started](https://docs.ultralytics.com/modes/#introduction)  
- [train](https://docs.ultralytics.com/modes/train/)
- [val](https://docs.ultralytics.com/modes/val/)
- [predict](https://docs.ultralytics.com/modes/predict/)
- [export](https://docs.ultralytics.com/modes/export/)

## Video Streaming
[Video streaming code](videostreaming/README.md)

## Tutorials and HOWTOs
[Computer vision tutorials](docs/README.md)

## Guides
- [Python environment setup](docs/pyenv_setup.md)
- [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/)
- [Adding your work](docs/adding_your_work.md)

