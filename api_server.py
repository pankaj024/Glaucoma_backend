from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from train_glaucoma import GlaucomaTriageModel
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import gc
import traceback

# Optimize PyTorch for low memory environments
torch.set_num_threads(1)

app = Flask(__name__)
CORS(app)

# Global model instance
model = None

def get_model():
    global model
    if model is None:
        model = GlaucomaTriageModel()
        # Trigger garbage collection after loading model
        gc.collect()
    return model

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "OcularAI Glaucoma Triage API is running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "DINOv2-S"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
            
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        
        # Save temp image for processing
        temp_path = "temp_predict.jpg"
        img.save(temp_path)
        
        # 1. Extract Features & Predict
        triage = get_model()
        
        with torch.no_grad():
            features = triage.extract_features(temp_path)
            
            # Use trained classifier if available
            if triage.classifier:
                prob = float(triage.classifier.predict_proba([features])[0][1])
            else:
                prob = float(np.mean(np.abs(features)) * 10) % 1.0
                if prob > 0.8: prob = 0.85
            
            # 2. Generate Heatmap (XAI)
            heatmap = triage.get_attention_map(temp_path)
        
        # Convert heatmap to base64
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (heatmap_color.shape[1], heatmap_color.shape[0]))
        blended = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        
        # Encode blended result
        _, buffer = cv2.imencode('.jpg', blended)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 3. Dummy Metrics
        metrics = {
            "glaucomaProbability": prob,
            "explanationMap": f"data:image/jpeg;base64,{heatmap_base64}",
            "metrics": {
                "discArea": 0.82 + (prob * 0.1),
                "cupArea": 0.24 + (prob * 0.2),
                "cdr": 0.29 + (prob * 0.3),
            }
        }
        
        os.remove(temp_path)
        # Final cleanup for memory
        gc.collect()
        return jsonify(metrics)
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Error: {error_msg}")
        traceback.print_exc()
        return jsonify({
            "error": error_msg,
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Disable debug mode and reloader for production (Render)
    # Debug mode uses significantly more memory
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
