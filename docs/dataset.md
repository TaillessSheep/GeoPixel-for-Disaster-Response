# Prepare Dataset 🚀

## 𝗚𝗲𝗼𝗣𝗶𝘅𝗲𝗹𝗗 𝗗𝗮𝘁𝗮𝘀𝗲𝘁 📂: 

GeoPixelD is a large-scale, grounded conversation dataset designed for precise object-level description and understanding. It contains over 53,000 phrases linked to more than 600,000 objects, enabling fine-grained multimodal grounding.

## 💻 Download GeoPixelD 📊 

- Download Annotations from [MBZUAI/GeoPixelD](https://huggingface.co/datasets/MBZUAI/GeoPixelD)
- Images -> [Download](https://captain-whu.github.io/DOTA/index.html). 
- GeoPixelD uses [iSAID](https://captain-whu.github.io/iSAID/dataset.html) Images which are the same as the DOTA-v1.0 dataset.
- Prepare the data using the [iSAID Development Kit](https://github.com/CAPTAIN-WHU/iSAID_Devkit) 
  - Split the training and validation images into 800 × 800 pixel patches, then move the training images to the 'train' folder and the validation images to the 'test' folder of GeoPixelD.
  - Place them in same folder as annotations. The final dataset should follow this structure:
    ```
        GeoPixelD
        ├── test
        │       P0003_0_800_347_1147.json
        │       P0003_0_800_347_1147.png
        │       P0003_223_1023_0_800.json
        │       P0003_223_1023_0_800.png
        │       ...
        ├── train
        │       P0224_0_800_0_800.json
        │       P0224_0_800_0_800.png
        │       P0224_0_800_600_1400.json
        │       P0224_0_800_600_1400.png
        │       ...
        GeoPixelD.json
    ```
