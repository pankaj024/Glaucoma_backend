import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import requests
import zipfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pickle

# DINOv2 Model - Small version for speed and scalability
DINOV2_MODEL = "dinov2_vits14"

class GlaucomaTriageModel:
    def __init__(self, model_path='glaucoma_classifier.pkl'):
        print(f"Loading {DINOV2_MODEL}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Completely bypass GitHub API to avoid 'Authorization' (rate limit) errors on Render
        try:
            # Attempt to load using the torch.hub.load with source='github' but forcing no API check
            # if possible, otherwise we manually download or use a direct torch.load if we had the weights.
            # However, the most reliable way to bypass the 'Authorization' error is to 
            # ensure we don't trigger the GitHub API request for the version check.
            self.model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL, trust_repo=True, skip_validation=True).to(self.device)
        except Exception as e:
            print(f"Primary load failed: {e}. Trying absolute offline fallback...")
            # If the above still fails due to GitHub API, we try to load from the local cache directory directly
            # by pointing to the hub folder
            hub_dir = torch.hub.get_dir()
            model_dir = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
            if os.path.exists(model_dir):
                self.model = torch.hub.load(model_dir, DINOV2_MODEL, source='local', trust_repo=True).to(self.device)
            else:
                # Last resort: try loading without validation
                self.model = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL, trust_repo=True, force_reload=False).to(self.device)
            
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.classifier = None # Linear Probe
        self.model_path = model_path
        
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)
                print("Loaded existing classifier.")

    def extract_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_t)
        return features.cpu().numpy().flatten()

    def get_attention_map(self, image_path):
        """
        Extracts self-attention maps from the last layer of DINOv2.
        This is the core 'Explainable AI' feature.
        """
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get attention weights using a hook
        # DINOv2 ViT hub model doesn't have get_last_selfattention
        last_attn = self.model.blocks[-1].attn
        attn_weights = []

        def hook_fn(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            
            q = q * module.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn_weights.append(attn)

        handle = last_attn.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            self.model(img_t)
        
        handle.remove()
        
        if not attn_weights:
            return np.zeros((h, w))
            
        # Reshape to (heads, patches, patches)
        # Patch size is 14x14, so for 224x224 we have 16x16 = 256 patches + 1 [CLS] token
        attentions = attn_weights[0]
        nh = attentions.shape[1] # number of heads
        
        # We focus on the [CLS] token's attention to other patches
        # Shape: (heads, patches)
        cls_attn = attentions[0, :, 0, 1:]
        
        # Reshape to 16x16 grid
        grid_size = int(np.sqrt(cls_attn.shape[-1]))
        cls_attn = cls_attn.reshape(nh, grid_size, grid_size)
        
        # Average across heads or pick the most informative one
        avg_attention = cls_attn.mean(0).cpu().numpy()
        
        # Upscale to original image size
        avg_attention = cv2.resize(avg_attention, (w, h))
        return avg_attention

    def train(self, data_dir):
        """
        Trains a Linear Probe on the extracted DINOv2 features.
        """
        print("Starting Linear Probing...")
        X, y = [], []
        
        # Expected structure: data_dir/normal, data_dir/glaucoma
        for label, folder in enumerate(['normal', 'glaucoma']):
            path = os.path.join(data_dir, folder)
            if not os.path.exists(path): 
                print(f"Directory {path} not found.")
                continue
            
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            # Fast training for demo: use first 20 images
            files = files[:20]
            for img_name in tqdm(files, desc=f"Extracting {folder}"):
                feat = self.extract_features(os.path.join(path, img_name))
                X.append(feat)
                y.append(label)
        
        if not X:
            print("No data found for training.")
            return 0
            
        X = np.array(X)
        y = np.array(y)
        
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.classifier.fit(X, y)
        
        preds = self.classifier.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Training Accuracy: {acc * 100:.2f}%")
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
            print(f"Saved classifier to {self.model_path}")
            
        return acc

if __name__ == "__main__":
    triage = GlaucomaTriageModel()
    triage.train('data')

