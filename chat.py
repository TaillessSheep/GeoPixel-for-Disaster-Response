import os
import re
import sys
import cv2
import torch
import random
import argparse
import numpy as np
import transformers 
from model.geopixel import GeoPixelForCausalLM

def rgb_color_text(text, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def load_model(version='MBZUAI/GeoPixel-7B'):
    """
    Load the GeoPixel model and tokenizer.
    
    Args:
        version: Model version to load (default: 'MBZUAI/GeoPixel-7B')
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f'Initializing tokenizer from: {version}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx, bop_token_idx, eop_token_idx = [
        tokenizer(token, add_special_tokens=False).input_ids[0] for token in ['[SEG]','<p>', '</p>']
    ]
   
    kwargs = {"torch_dtype": torch.bfloat16}    
    geo_model_args = {
        "vision_pretrained": 'facebook/sam2-hiera-large',
        "seg_token_idx": seg_token_idx, # segmentation token index
        "bop_token_idx": bop_token_idx, # begining of phrase token index
        "eop_token_idx": eop_token_idx  # end of phrase token index
    }
    
    print(f'Loading model from: {version}')
    model = GeoPixelForCausalLM.from_pretrained(
        version, 
        low_cpu_mem_usage=True, 
        **kwargs,
        **geo_model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer
    
    model = model.bfloat16().cuda().eval()
    print('Model loaded and ready for inference')
    
    return model, tokenizer

def process_masks(pred_masks, image_path, response, vis_save_path="./vis_output"):
    """
    Process segmentation masks and save masked image.
    
    Args:
        pred_masks: Predicted masks from model (numpy array)
        image_path: Path to the original image
        response: Model response text
        vis_save_path: Directory to save masked images (default: "./vis_output")
        
    Returns:
        dict: {
            'has_masks': True,
            'masked_image_path': path to saved image,
            'phrases': list of phrases
        }
    """
    pred_masks = pred_masks[0]
    pred_masks = pred_masks.detach().cpu().numpy()
    pred_masks = pred_masks > 0
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    save_img = image_np.copy()
    pattern = r'<p>(.*?)</p>\s*\[SEG\]'
    matched_text = re.findall(pattern, response)
    phrases = [text.strip() for text in matched_text]

    for i in range(pred_masks.shape[0]):
        mask = pred_masks[i]
        color = [random.randint(0, 255) for _ in range(3)]
        mask_rgb = np.stack([mask, mask, mask], axis=-1) 
        color_mask = np.array(color, dtype=np.uint8) * mask_rgb

        save_img = np.where(mask_rgb, 
                (save_img * 0.5 + color_mask * 0.5).astype(np.uint8), 
                save_img)
    
    # Save masked image
    os.makedirs(vis_save_path, exist_ok=True)
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    save_path = f"{vis_save_path}/{os.path.basename(image_path).split('.')[0]}_masked.jpg"
    cv2.imwrite(save_path, save_img)
    
    return {
        'has_masks': True,
        'masked_image_path': save_path,
        'phrases': phrases
    }

def process_query_image(model, tokenizer, query, image_path, vis_save_path="./vis_output", max_new_tokens=300):
    """
    Process a query and image using the GeoPixel model.
    
    Args:
        model: Loaded GeoPixel model
        tokenizer: Loaded tokenizer
        query: Text query/prompt
        image_path: Path to image file
        vis_save_path: Directory to save visualization outputs (default: "./vis_output")
        max_new_tokens: Maximum number of tokens to generate (default: 300)
        
    Returns:
        dict: {
            'response': model response text,
            'has_masks': bool,
            'masked_image_path': path to masked image (if masks present),
            'phrases': list of phrases (if masks present)
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = [image_path]
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        response, pred_masks = model.evaluate(tokenizer, query, images=image, max_new_tokens=max_new_tokens)
    
    result = {
        'response': response.replace("\n", " ").replace("  ", " ").strip(),
        'has_masks': False
    }
    
    # Process masks if present
    if pred_masks and '[SEG]' in response:
        mask_result = process_masks(pred_masks, image_path, response, vis_save_path)
        result.update(mask_result)
    
    return result

def parse_args(args):
    parser = argparse.ArgumentParser(description="Chat with GeoPixel")
    parser.add_argument("--version", default="MBZUAI/GeoPixel-7B")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)

    os.makedirs(args.vis_save_path, exist_ok=True)

    # Load model using common function
    model, tokenizer = load_model(args.version)

    while True:
        query = input("Please input your query: ")
        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        # Process query and image using common function
        result = process_query_image(model, tokenizer, query, image_path, args.vis_save_path)
        
        # Display results
        if result['has_masks']:
            # Reconstruct description with colored phrases for terminal display
            response = result['response']
            phrases = result['phrases']
            split_desc = response.split('[SEG]')
            cleaned_segments = [re.sub(r'<p>(.*?)</p>', '', part).strip() for part in split_desc]
            reconstructed_desc = ""
            for i, part in enumerate(cleaned_segments):
                reconstructed_desc += part + ' '
                if i < len(phrases):
                    # Color the phrases for terminal display
                    color = [random.randint(0, 255) for _ in range(3)]
                    colored_phrase = rgb_color_text(phrases[i], color[0], color[1], color[2])
                    reconstructed_desc += colored_phrase + ' '    
            print(reconstructed_desc)
            print("{} has been saved.".format(result['masked_image_path']))
        else:
            print(result['response'])

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
