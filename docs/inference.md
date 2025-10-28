# Inference Guide 
GeoPixel-7B is a powerful model for pixel-grounded analysis of images. Follow the steps below to interact with it via chat.py for detailed descriptions and segmentation mask generation.

## Grounded Remote Sensing Image Analysis
To chat with GeoPixel-7B to obtain detailed pixel grounded analysis of image use following command:

```
CUDA_VISIBLE_DEVICES=0 python chat.py --version='MBZUAI/GeoPixel-7B'
```

Input the text prompt and then the image path. For example，
### Example Usage
```
- Please input your prompt: Can you provide a thorough description of this image? Please output with interleaved segmentation masks for the corresponding phrases.
- Please input the image path: images/example1.png
```
## Referring Remote Sensing Image Segmentation (RRSIS)
To obtain grounded segmentation masks for specific referring phrases in remote sensing images, use GeoPixel-7B-RES:

```
CUDA_VISIBLE_DEVICES=0 python chat.py --version='MBZUAI/GeoPixel-7B-RES'
```
Usage Instructions

Input a text prompt specifying the referring phrase ({ref_exp}) in 

> Could you provide a segmentation mask for {ref_exp} in this image?

and provide the path to the image.

### Example Usage

```
- Please input your prompt: Could you provide a segmentation mask for {ref_exp} in this image?
- Please input the image path: images/example1-RES.jpg
```
ref_exp examples for images/example1-RES.jpg: 
-  "red car",
-  "white car at right"
-  "tree at bottom right"
-  "a vehicle on the left"
-  "house at bottom right"
-  "purple tree at bottom"

```
- Please input your prompt: Could you provide a segmentation mask for red car in this image?
- Please input the image path: images/example1-RES.jpg
```
