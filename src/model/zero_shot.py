import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, pipeline
import torchvision.transforms as transforms
import onnxruntime as ort
import os
from tqdm import tqdm

class ZeroShotModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.custom_categories = []

    def load_model(self):
        # Load the pre-trained zero-shot model here
        pass

    def set_custom_categories(self, categories: list):
        self.custom_categories = categories

    def process_frame(self, frame):
        # Process the input frame and generate predictions
        predictions = []
        # Implement the logic for predictions using the model
        return predictions

    def annotate_frame(self, frame, predictions):
        # Annotate the frame with bounding boxes and labels
        for prediction in predictions:
            # Draw bounding boxes and labels on the frame
            pass
        return frame

class ZeroShotDetector:
    def __init__(self, device="cpu", cache_dir=None):
        self.device = device
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Disable HF warning about symlinks
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"
        
        print("Loading CLIP model (this may take a while the first time)...")
        model_name = "openai/clip-vit-large-patch14"
        
        self.model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32
        ).to(device)
        
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print("Model loaded successfully!")

    def predict(self, image, categories, confidence_threshold=0.15):
        # Add better prompts for detection
        prompts = [
            f"a clear photo of a {cat}, high quality, centered" 
            for cat in categories
        ]
        
        # Process inputs
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get probability scores
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=0)
        
        # Format predictions
        predictions = []
        for category, confidence in zip(categories, probs.cpu().numpy()):
            if confidence > confidence_threshold:
                predictions.append({
                    'category': category,
                    'confidence': float(confidence)
                })
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)