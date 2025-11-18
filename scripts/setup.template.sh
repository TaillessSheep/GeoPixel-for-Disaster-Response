# Load slurm module
module load slurm

# Run container and save after setup
srun --account=mscaisuperpod --partition=normal --gpus=1 \
     --container-image="docker://nvcr.io#nvidia/pytorch:24.03-py3" \
     --container-mounts=/home/[user]/[path]/GeoPixel-for-Disaster-Response:/workspace \
     --no-container-mount-home --container-remap-root --container-writable \
     --container-save="/home/$USER/containers/geopixel.sqsh" \
     --pty bash -c "
cd /workspace
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.6.3 --no-build-isolation
pip install -r requirements.txt
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
python -m pip install --no-cache-dir opencv-python-headless==4.8.0.74
bash"