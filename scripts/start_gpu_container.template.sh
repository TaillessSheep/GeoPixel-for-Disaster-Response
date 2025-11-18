#!/bin/bash

srun --account=mscaisuperpod --partition=normal --gpus=1 \
     --container-image="/home/$USER/containers/geopixel.sqsh" \
     --container-mounts=/home/[user]/[path]/GeoPixel-for-Disaster-Response:/workspace \
     --no-container-mount-home --container-remap-root --container-writable \
     --pty bash
