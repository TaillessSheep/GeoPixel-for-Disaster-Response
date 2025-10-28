# 🛠️ Installation

Set up conda envirnment:

```
conda create --name=geopixel python=3.10
conda activate geopixel

git clone https://github.com/mbzuai-oryx/GeoPixel.git
cd GeoPixel

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.6.3 --no-build-isolation

pip install -r requirements.txt

```
