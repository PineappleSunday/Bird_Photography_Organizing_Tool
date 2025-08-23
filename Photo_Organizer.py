import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil

def classify_and_organize_photos(source_folder, destination_folder):
    """
    Classifies all images in a source folder and organizes them into
    species-named folders in a destination folder.
    """
    # Step 1: Load the pre-trained model and class labels
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # You must have this file in the same directory as your script
    with open("imagenet_classes.txt", "r", encoding="utf-8") as f:
        categories = [s.strip() for s in f.readlines()]
        
    # Step 2: Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Step 3: Iterate through all images in the source folder
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(source_folder, filename)
            
            try:
                # Load the image
                input_image = Image.open(image_path).convert('RGB')
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0)
                
                # Make the prediction
                with torch.no_grad():
                    output = model(input_batch)
                
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_catid = torch.topk(probabilities, 1)
                
                # Get the predicted species name
                predicted_species = categories[top_catid[0].item()]
                
                # Sanitize the name for use as a folder name
                folder_name = predicted_species.replace(" ", "_")
                species_dir = os.path.join(destination_folder, folder_name)
                
                # Create the species folder and move the image
                if not os.path.exists(species_dir):
                    os.makedirs(species_dir)
                
                shutil.move(image_path, os.path.join(species_dir, filename))
                print(f"Moved '{filename}' to '{folder_name}' folder.")
            
            except Exception as e:
                print(f"Could not process '{filename}'. Error: {e}")

if __name__ == '__main__':
    # Define your source and destination folders
    source_folder = "G:/Programming Projects/Photos for Test"
    destination_folder = "G:/Programming Projects/Organized Photos"
    
    classify_and_organize_photos(source_folder, destination_folder)