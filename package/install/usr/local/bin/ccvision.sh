#!/bin/sh

dt=date
echo "ccvision `$dt`" >> /var/log/ccvision.log

su - jetson

export PYTHONPATH=/home/jetson/ComputerVision
python -m ccvision.main -c /home/jetson/ComputerVision/config.yaml
