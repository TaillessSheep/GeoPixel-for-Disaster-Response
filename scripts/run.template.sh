#!/bin/bash

cd [path]
# e.g. cd ~/5203/GeoPixel-for-Disaster-Response

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.6.3 --no-build-isolation
pip install -r requirements.txt

python -m pip uninstall -y opencv-python opencv-contrib-python     opencv-python-headless opencv-contrib-python-headless
python -m pip install --no-cache-dir opencv-python-headless==4.8.0.74 

CUDA_VISIBLE_DEVICES=0 python chat.py --version='MBZUAI/GeoPixel-7B'