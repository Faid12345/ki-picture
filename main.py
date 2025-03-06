import os
import requests
from datetime import datetime
import random
from flask import jsonify

# Free Hugging Face model for image generation
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # This model works better with the free API
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Token from Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")  # Store your Hugging Face Token in Vercel's environment

def generate_image(request):
    prompt = request.json.get('prompt', '')
    style = request.json.get('style', 'standard')
    size = request.json.get('size', '512x512')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        # Parse size
        width, height = map(int, size.split('x'))
        
        # Adjust the style
        style_prompts = {
            "standard": "",
            "photo": "ultra realistic photograph, 4k, highly detailed, professional photography, ",
            "3d": "3D render, octane render, cinema 4D, blender, highly detailed 3D model, ",
            "drawing": "detailed drawing, illustration, sketch, pen and ink, "
        }
        enhanced_prompt = style_prompts[style] + prompt

        # Negative prompts for style
        negative_prompts = {
            "standard": "blurry, bad quality, distorted, ugly, low resolution, pixelated, disfigured faces, unrealistic proportions",
            "photo": "drawing, painting, illustration, 3d render, cartoon, anime, sketch, digital art, blurry, bad quality",
            "3d": "photograph, 2D, flat, drawing, painting, sketch, cartoon, anime, blurry, bad quality",
            "drawing": "photograph, 3d render, bad drawing, blurry, bad quality, realistic, photorealistic"
        }
        
        # Call Hugging Face API
        payload = {
            "inputs": enhanced_prompt,
            "negative_prompt": negative_prompts[style],
            "wait_for_model": True,
            "parameters": {
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "scheduler": "DPMSolverMultistep",
                "width": width,
                "height": height
            }
        }
        
        # Make the request
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to generate image'}), 500
        
        # Save the image to static folder
        image_bytes = response.content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = random.randint(1000, 9999)
        filename = f"generated_{timestamp}_{random_id}.png"
        filepath = f"/static/images/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        return jsonify({
            'image_url': filepath,
            'prompt': prompt,
            'is_ai': True
        })

