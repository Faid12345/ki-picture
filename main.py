
from flask import Flask, render_template, request, jsonify
import requests
import os
import random
import json
import io
import base64
from datetime import datetime
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Free Hugging Face model for image generation
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # This model works better with the free API
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# You'll need to set this in your environment or secrets
# Get it from https://huggingface.co/settings/tokens
HF_TOKEN = "HF_token"  # Direct token assignment for all users

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    prompt = request.json.get('prompt', '')
    style = request.json.get('style', 'standard')
    size = request.json.get('size', '512x512')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Always use the admin token defined in the environment
    global HF_TOKEN
    
    print(f"Using admin token for all requests")

    try:
        # First, try using Hugging Face's API
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            if not HF_TOKEN:
                # No token provided, go directly to fallback
                raise Exception("No Hugging Face token provided")
            
            # Parse size to width and height
            width, height = map(int, size.split('x'))
            
            # Adjust prompt based on selected style
            style_prompts = {
                "standard": "",
                "photo": "ultra realistic photograph, 4k, highly detailed, professional photography, ",
                "3d": "3D render, octane render, cinema 4D, blender, highly detailed 3D model, ",
                "drawing": "detailed drawing, illustration, sketch, pen and ink, "
            }
            
            # Add style to the prompt
            enhanced_prompt = style_prompts[style] + prompt
            
            # Adjust negative prompts based on style
            negative_prompts = {
                "standard": "blurry, bad quality, distorted, ugly, low resolution, pixelated, disfigured faces, unrealistic proportions",
                "photo": "drawing, painting, illustration, 3d render, cartoon, anime, sketch, digital art, blurry, bad quality",
                "3d": "photograph, 2D, flat, drawing, painting, sketch, cartoon, anime, blurry, bad quality",
                "drawing": "photograph, 3d render, bad drawing, blurry, bad quality, realistic, photorealistic"
            }
                
            # Make a direct API call to Hugging Face
            payload = {
                "inputs": enhanced_prompt,
                "negative_prompt": negative_prompts[style],
                "wait_for_model": True,
                "parameters": {
                    "guidance_scale": 7.5,  # Higher values create more realistic images (range 1-20)
                    "num_inference_steps": 50,  # More steps = higher quality (but slower)
                    "scheduler": "DPMSolverMultistep",  # Better scheduler for realistic results
                    "width": width,
                    "height": height
                }
            }
            
            # Call the Hugging Face API directly
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"API returned error {response.status_code}: {response.text}")
                
            # Get the image bytes
            image_bytes = response.content
            
            # Save the image to a temporary file
            if not os.path.exists('static'):
                os.makedirs('static')
            if not os.path.exists('static/images'):
                os.makedirs('static/images')
                
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_id = random.randint(1000, 9999)
            filename = f"generated_{timestamp}_{random_id}.png"
            filepath = os.path.join('static/images', filename)
            
            print(f"Generated image with prompt: '{prompt}'")
            
            # Save the image
            with open(filepath, "wb") as f:
                f.write(image_bytes)
                
            # Create a URL to the image
            image_url = f"/static/images/{filename}"
            
            print(f"Successfully generated AI image: {image_url}")
            
            return jsonify({
                'image_url': image_url,
                'prompt': prompt,
                'is_ai': True
            })
            
        except Exception as ai_error:
            print(f"AI generation failed: {str(ai_error)}")
            
            # Fallback to stock photos if AI generation fails
            search_term = prompt.replace(' ', '+')
            
            # Use fixed dimensions to ensure more reliable responses
            width = 500
            height = 500
            
            # Map common search terms to known good Pexels IDs (fallback)
            pexels_mappings = {
                'dog': '1108099',
                'cat': '617278',
                'mountain': '1366909',
                'beach': '1005417',
                'city': '466685',
                'forest': '15286',
                'flower': '736230',
                'food': '1099680',
                'car': '210019'
            }
            
            # Check if the search term contains any of our mapped keywords
            found_mapping = False
            search_term_clean = search_term.lower()
            for key, value in pexels_mappings.items():
                if key in search_term_clean:
                    image_url = f"https://images.pexels.com/photos/{value}/pexels-photo-{value}.jpeg?auto=compress&cs=tinysrgb&w=500"
                    found_mapping = True
                    break
            
            # If no mapping found, use a fallback method
            if not found_mapping:
                # Use Loremflickr as last resort
                timestamp = int(datetime.now().timestamp())
                image_url = f"https://loremflickr.com/{width}/{height}/{search_term}?lock={timestamp}"
            
            print(f"Falling back to stock photo: {image_url}")
            
            return jsonify({
                'image_url': image_url,
                'prompt': prompt,
                'is_ai': False
            })
            
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/images', methods=['GET'])
def get_images():
    # In a real app, you would retrieve images from a database
    # This is just a placeholder that returns some sample data
    sample_images = [
        {
            'id': 1,
            'prompt': 'Beautiful mountain landscape',
            'image_url': 'https://images.pexels.com/photos/1366909/pexels-photo-1366909.jpeg?auto=compress&cs=tinysrgb&w=500',
            'is_ai': False
        },
        {
            'id': 2,
            'prompt': 'Futuristic city skyline',
            'image_url': 'https://images.pexels.com/photos/466685/pexels-photo-466685.jpeg?auto=compress&cs=tinysrgb&w=500',
            'is_ai': False
        },
        {
            'id': 3,
            'prompt': 'Cute dog',
            'image_url': 'https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=500',
            'is_ai': False
        }
    ]
    return jsonify(sample_images)

if __name__ == '__main__':
    # Create necessary directories
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    app.run(host='0.0.0.0', port=8080)
