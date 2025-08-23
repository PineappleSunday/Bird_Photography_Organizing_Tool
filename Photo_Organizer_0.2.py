import os
import shutil
import json
import rawpy
import torch
from transformers import pipeline
from PIL import Image
import numpy as np

def get_name_mapping(json_file_path):
    """
    Reads a JSON file and returns a dictionary mapping scientific names to common names.
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            scientific_names = data['name']
            common_names = data['common_name']
            
            mapping = {scientific_names[str(i)]: common_names[str(i)] for i in range(len(scientific_names))}
            return mapping
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found. Cannot create name mapping.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return None

def classify_and_organize_photos(source_folder, destination_folder, name_mapping):
    """
    Classifies all images in a source folder and organizes them into
    common-name folders in a destination folder based on a confidence threshold.
    """
    CONFIDENCE_THRESHOLD = 0.50
    RAW_FILE_EXTENSIONS = ('.nef', '.cr2', '.arw', '.ORF', '.orf') # Add other raw formats here

    if not os.path.exists(source_folder):
        print(f"Error: The source folder '{source_folder}' does not exist.")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    unclassified_dir = os.path.join(destination_folder, "Unclassified")
    if not os.path.exists(unclassified_dir):
        os.makedirs(unclassified_dir)

    print("Loading the iNaturalist classification model...")
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1

    print("Loading the iNaturalist classification model...")
    try:
        model_name = "timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21"
        classifier = pipeline("image-classification", model=model_name, device=device)
        print("Model loaded successfully on GPU." if device == 0 else "Model loaded successfully on CPU.")
    except Exception as e:
        print(f"Error loading the model. Error: {e}")
        return

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.jpg', '.jpeg') + RAW_FILE_EXTENSIONS):
            try:
                # Handle raw files
                if filename.lower().endswith(RAW_FILE_EXTENSIONS):
                    with rawpy.imread(file_path) as raw:
                        # Process raw data into an RGB image
                        rgb = raw.postprocess(
                            gamma=(2.2, 4.5),
                            no_auto_bright=True,
                            use_camera_wb=True
                        )
                        # Create an in-memory Pillow image from the processed data
                        input_image = Image.fromarray(rgb)
                else:
                    # Handle JPEG files
                    input_image = Image.open(file_path).convert('RGB')
                
                # The rest of the classification process is the same
                predictions = classifier(input_image)
                
                if predictions:
                    top_prediction = predictions[0]
                    predicted_species_scientific = top_prediction['label']
                    confidence_score = top_prediction['score']
                    
                    if confidence_score >= CONFIDENCE_THRESHOLD:
                        predicted_species_common = name_mapping.get(predicted_species_scientific, predicted_species_scientific)
                        folder_name = predicted_species_common.replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")
                        species_dir = os.path.join(destination_folder, folder_name)
                        
                        if not os.path.exists(species_dir):
                            os.makedirs(species_dir)
                        
                        shutil.move(file_path, os.path.join(species_dir, filename))
                        print(f"Moved '{filename}' to '{predicted_species_common}' folder with confidence: {confidence_score:.4f}")
                    else:
                        shutil.move(file_path, os.path.join(unclassified_dir, filename))
                        print(f"Moved '{filename}' to 'Unclassified' folder due to low confidence ({confidence_score:.4f}).")
                else:
                    shutil.move(file_path, os.path.join(unclassified_dir, filename))
                    print(f"Could not classify '{filename}'. Moved to 'Unclassified' folder.")

            except Exception as e:
                print(f"Could not process '{filename}'. Error: {e}. Moved to 'Unclassified' folder.")
                shutil.move(file_path, os.path.join(unclassified_dir, filename))

if __name__ == '__main__':
    taxonomy_file_path = r"G:\Programming Projects\bird_classes.json" 
    
    name_mapping_dict = get_name_mapping(taxonomy_file_path)
    
    if name_mapping_dict:
        source_folder = r"G:\Programming Projects\Photos for Test"
        destination_folder = r"G:\Programming Projects\Organized Photos"
        
        classify_and_organize_photos(source_folder, destination_folder, name_mapping_dict)
    else:
        print("Failed to load name mapping. Script terminated.")