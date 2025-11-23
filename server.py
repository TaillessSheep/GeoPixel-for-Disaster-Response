#!/usr/bin/env python3
"""
Simple HTTP server with health check endpoint and chat API for GeoPixel.
Requires Flask: pip install flask
"""

import sys
import os
import re
import cv2
import torch
import random
import argparse
import numpy as np
import transformers
from datetime import datetime
try:
    from datetime import timezone
    HAS_TIMEZONE = True
except ImportError:
    HAS_TIMEZONE = False
from model.geopixel import GeoPixelForCausalLM

try:
    from flask import Flask, jsonify, request
except ImportError:
    print("Error: Flask is not installed. Please install it with: pip install flask")
    sys.exit(1)

try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
    print("Warning: flask-cors not installed. CORS headers will be set manually.")
    print("For better CORS support, install: pip install flask-cors")

app = Flask(__name__)

# Enable CORS for all routes
if HAS_CORS:
    CORS(app)
else:
    # Manual CORS headers as fallback
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

# Global variables for model and tokenizer
model = None
tokenizer = None

# Handle CORS preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
        return response

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    Returns server status and timestamp.
    """
    # Use timezone-aware datetime if available, otherwise fall back to utcnow()
    if HAS_TIMEZONE:
        timestamp = datetime.now(timezone.utc).isoformat()
    else:
        timestamp = datetime.utcnow().isoformat() + 'Z'
    
    return jsonify({
        'status': 'healthy',
        'timestamp': timestamp,
        'service': 'GeoPixel Server'
    }), 200

@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with basic server information.
    """
    return jsonify({
        'message': 'GeoPixel Server is running',
        'endpoints': {
            'health': '/health',
            'upload': '/upload (POST)',
            'chat': '/chat (POST)'
        }
    }), 200

@app.route('/vis_output/<path:filename>', methods=['GET'])
def serve_vis_output(filename):
    """
    Serve visualization output files (masked images).
    """
    from flask import send_from_directory
    vis_dir = os.path.abspath('./vis_output')
    return send_from_directory(vis_dir, filename)

@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_uploads(filename):
    """
    Serve uploaded files.
    """
    from flask import send_from_directory
    upload_dir = os.path.abspath('./uploads')
    return send_from_directory(upload_dir, filename)

@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload endpoint that accepts image files and saves them to the server.
    
    Expected: multipart/form-data with 'file' field
    
    Returns:
    {
        "path": "path/to/saved/image.tif",
        "filename": "image.tif"
    }
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create uploads directory if it doesn't exist
        upload_dir = './uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file
        filename = file.filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(upload_dir, safe_filename)
        file.save(file_path)
        
        # Return the absolute path
        abs_path = os.path.abspath(file_path)
        
        return jsonify({
            'path': abs_path,
            'filename': safe_filename
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error uploading file: {str(e)}'
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint that processes prompts with images using the GeoPixel model.
    
    Expected: multipart/form-data with 'file' and 'prompt' fields
    
    Returns:
    {
        "response": "model response text",
        "has_masks": true/false,
        "masked_image_path": "path/to/saved/masked/image.jpg" (if masks present)
    }
    """
    if model is None or tokenizer is None:
        return jsonify({
            'error': 'Model not loaded. Server may still be initializing.'
        }), 503
    
    try:
        # Require file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided. Send multipart/form-data with file and prompt fields'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        query = request.form.get('prompt', '')
        if not query:
            return jsonify({'error': 'Missing required field: prompt'}), 400
        
        # Save uploaded file temporarily
        upload_dir = './uploads'
        os.makedirs(upload_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, safe_filename)
        file.save(file_path)
        image_path = os.path.abspath(file_path)
        
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file not found: {image_path}'}), 404
        
        # Process the request
        image = [image_path]
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            response, pred_masks = model.evaluate(tokenizer, query, images=image, max_new_tokens=300)
        
        result = {
            'response': response.replace("\n", " ").replace("  ", " ").strip(),
            'has_masks': False
        }
        
        # Process masks if present
        if pred_masks and '[SEG]' in response:
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
            vis_save_path = "./vis_output"
            os.makedirs(vis_save_path, exist_ok=True)
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            save_path = f"{vis_save_path}/{os.path.basename(image_path).split('.')[0]}_masked.jpg"
            cv2.imwrite(save_path, save_img)
            
            result['has_masks'] = True
            result['masked_image_path'] = save_path
            result['phrases'] = phrases
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing request: {str(e)}'
        }), 500

def load_model(version='MBZUAI/GeoPixel-7B'):
    """
    Load the GeoPixel model and tokenizer.
    This is called once on server startup.
    """
    global model, tokenizer
    
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
        "seg_token_idx": seg_token_idx,
        "bop_token_idx": bop_token_idx,
        "eop_token_idx": eop_token_idx
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start GeoPixel server")
    parser.add_argument('--host', default='0.0.0.0', type=str, help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', default=5000, type=int, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--version', default='MBZUAI/GeoPixel-7B', type=str, help='Model version to load (default: MBZUAI/GeoPixel-7B)')
    return parser.parse_args()

def main():
    """Main function to start the server."""
    args = parse_args()
    
    # Load model on startup
    print("Loading GeoPixel model...")
    try:
        load_model(args.version)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Server will start but /chat endpoint will not be available.")
        sys.exit(1)
    
    print(f"\nStarting GeoPixel server on {args.host}:{args.port}")
    print(f"Health check endpoint: http://{args.host}:{args.port}/health")
    print(f"Chat endpoint: http://{args.host}:{args.port}/chat")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == '__main__':
    main()

