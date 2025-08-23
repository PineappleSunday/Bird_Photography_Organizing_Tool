# ML.py

import rawpy
import torch
from transformers import pipeline
import json
from PIL import Image
from io import StringIO
import sys

# A dictionary to translate scientific names to common names
# This is a placeholder and should be loaded from a file
name_mapping_dict = {}

def get_name_mapping(json_file_path):
    """
    Reads a JSON file and returns a dictionary mapping scientific names to common names.
    """
    global name_mapping_dict
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            scientific_names = data['name']
            common_names = data['common_name']
            name_mapping_dict = {scientific_names[str(i)]: common_names[str(i)] for i in range(len(scientific_names))}
            return True
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found. Cannot create name mapping.")
        return False
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return False

# Use a buffer to capture initial print statements
console_buffer = StringIO()
sys.stdout = console_buffer

# Load name mapping from your file
taxonomy_file_path = r"G:\Programming Projects\bird_classes.json"
if not get_name_mapping(taxonomy_file_path):
    print("Failed to load name mapping. The application will not be able to function correctly.")


# Abstract base class for models
class BaseModel:
    def __init__(self, model_name, model_description):
        self.model_name = model_name
        self.model_info = {"name": model_name, "description": model_description}
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        print(f"Loading model: {self.model_name}...")
        try:
            self.pipeline = pipeline("image-classification", model=self.model_name, device=self.device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.pipeline = None
            
    def is_loaded(self):
        return self.pipeline is not None

    def load_image_from_path(self, file_path):
        if file_path.lower().endswith(('.nef', '.cr2', '.arw', '.orf')):
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess()
                return Image.fromarray(rgb)
        else:
            return Image.open(file_path).convert('RGB')

# Bird Classification Model
class BirdClassifierModel(BaseModel):
    def __init__(self):
        super().__init__(
            model_name="timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21",
            model_description="A Vision Transformer model fine-tuned on the iNaturalist 2021 dataset (10,000 species) for high-accuracy bird classification."
        )
    
    def classify(self, image):
        if not self.is_loaded():
            raise RuntimeError("Classification model is not loaded.")
        return self.pipeline(image)

    def get_common_name(self, scientific_name):
        return name_mapping_dict.get(scientific_name, scientific_name)

# Object Detection Model
class ObjectDetectorModel(BaseModel):
    def __init__(self):
        super().__init__(
            model_name="facebook/detr-resnet-50",
            model_description="A fine-tuned DETR model for object detection. It can detect and locate multiple objects in an image."
        )

    def load_model(self):
        print(f"Loading object detection model: {self.model_name}...")
        try:
            self.pipeline = pipeline("object-detection", model=self.model_name, device=self.device)
            print("Object detection model loaded successfully.")
        except Exception as e:
            print(f"Error loading object detection model: {e}")
            self.pipeline = None
            
    def detect(self, image):
        if not self.is_loaded():
            raise RuntimeError("Object detection model is not loaded.")
        return self.pipeline(image)
