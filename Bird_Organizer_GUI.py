import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import os
import rawpy
from transformers import pipeline
import torch
import json
import numpy as np
import sys
from io import StringIO
import threading

# Redirect stdout to a custom buffer
class ConsoleRedirector:
    def __init__(self, text_widget, initial_output):
        self.text_widget = text_widget
        self.stdout = sys.stdout
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, initial_output)
        self.text_widget.configure(state='disabled')
        sys.stdout = self

    def write(self, text):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
        self.stdout.write(text)

    def flush(self):
        self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout

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

# Use a buffer to capture initial print statements
console_buffer = StringIO()
sys.stdout = console_buffer

# Load name mapping from your file
taxonomy_file_path = r"G:\Programming Projects\bird_classes.json"
name_mapping_dict = get_name_mapping(taxonomy_file_path)

if not name_mapping_dict:
    print("Failed to load name mapping. The application will now close.")
    sys.exit()

class BirdClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("Bird Photo Classifier")
        master.geometry("800x600")

        self.classifier = None
        self.model_info = {
            "name": "timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21",
            "description": "A Vision Transformer model fine-tuned on the iNaturalist 2021 dataset (10,000 species) for high-accuracy bird classification."
        }
        self.console_window = None

        self.create_menu()
        self.create_widgets()
        
        # Load model and print to buffer
        self.load_model()
        
        # Restore stdout after initial setup is complete
        sys.stdout = sys.__stdout__

    def create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Close", command=self.master.destroy)
        
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(
            label=f"Active Model: {self.model_info['name']}",
            command=self.show_model_info
        )
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Console", command=self.open_console)
        tools_menu.add_command(label="Object Detection", command=self.open_object_detection_gui)
        
    def create_widgets(self):
        self.image_label = tk.Label(self.master)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

        self.info_frame = tk.Frame(self.master)
        self.info_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.title_label = tk.Label(self.info_frame, text="Top 5 Predictions", font=("Helvetica", 16))
        self.title_label.pack(pady=(0, 10))

        self.predictions_text = tk.Label(self.info_frame, justify=tk.LEFT, font=("Helvetica", 12))
        self.predictions_text.pack()

        self.open_button = tk.Button(self.info_frame, text="Open Image", command=self.open_file)
        self.open_button.pack(pady=(20, 0))

    def open_console(self):
        if self.console_window is None or not self.console_window.winfo_exists():
            self.console_window = tk.Toplevel(self.master)
            self.console_window.title("Console Output")
            self.console_window.geometry("600x400")
            
            console_text = scrolledtext.ScrolledText(self.console_window, wrap=tk.WORD, state='normal')
            console_text.pack(expand=True, fill='both')
            
            self.redirector = ConsoleRedirector(console_text, console_buffer.getvalue())

    def open_object_detection_gui(self):
        # Open the object detection GUI in a new Toplevel window
        detection_window = tk.Toplevel(self.master)
        ObjectDetectionGUI(detection_window)

    def show_model_info(self):
        messagebox.showinfo("Model Information", self.model_info['description'])
        
    def load_model(self):
        if self.classifier:
            self.classifier = None
        
        print("Loading the classification model...")
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            self.classifier = pipeline("image-classification", model=self.model_info['name'], device=device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading the model: {e}")
            self.classifier = None
            
    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.nef *.cr2 *.arw *.orf")]
        )
        if file_path:
            self.display_image(file_path)
            self.classify_image(file_path)

    def display_image(self, file_path):
        try:
            if file_path.lower().endswith(('.nef', '.cr2', '.arw', '.orf')):
                with rawpy.imread(file_path) as raw:
                    rgb = raw.postprocess()
                    img = Image.fromarray(rgb)
            else:
                img = Image.open(file_path).convert('RGB')
                
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.image_label.config(text=f"Error loading image: {e}")
            self.image_label.image = None
            
    def classify_image(self, file_path):
        if not self.classifier:
            print("Classification Error: No model is currently loaded.")
            return

        try:
            if file_path.lower().endswith(('.nef', '.cr2', '.arw', '.orf')):
                with rawpy.imread(file_path) as raw:
                    rgb = raw.postprocess()
                    input_image = Image.fromarray(rgb)
            else:
                input_image = Image.open(file_path).convert('RGB')
            
            predictions = self.classifier(input_image)
            
            if predictions:
                predictions_str = ""
                for i in range(min(5, len(predictions))):
                    pred = predictions[i]
                    scientific_name = pred['label']
                    common_name = name_mapping_dict.get(scientific_name, scientific_name)
                    confidence = pred['score']
                    predictions_str += f"{i+1}. {common_name}\n   Confidence: {confidence:.4f}\n"
                self.predictions_text.config(text=predictions_str)
            else:
                self.predictions_text.config(text="Could not classify the image.")
        except Exception as e:
            print(f"Error classifying image: {e}")

class ObjectDetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Object Detection")
        master.geometry("800x600")

        self.detection_model = None
        self.model_info = {
            "name": "facebook/detr-resnet-50",
            "description": "A fine-tuned DETR model for object detection. It can detect and locate multiple objects in an image."
        }
        
        self.create_widgets()
        
        # Load model and run in a separate thread to prevent GUI from freezing
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        self.image_label = tk.Label(self.master)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

        self.info_frame = tk.Frame(self.master)
        self.info_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.title_label = tk.Label(self.info_frame, text="Detected Objects", font=("Helvetica", 16))
        self.title_label.pack(pady=(0, 10))

        self.predictions_text = tk.Label(self.info_frame, justify=tk.LEFT, font=("Helvetica", 12))
        self.predictions_text.pack()

        self.open_button = tk.Button(self.info_frame, text="Open Image", command=self.open_file)
        self.open_button.pack(pady=(20, 0))

    def load_model(self):
        print("\nLoading the object detection model. This may take a moment...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.detection_model = pipeline("object-detection", model=self.model_info['name'], device=device)
            print("Object detection model loaded successfully.")
        except Exception as e:
            print(f"Error loading the object detection model: {e}")
            self.detection_model = None

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.nef *.cr2 *.arw *.orf")]
        )
        if file_path:
            self.display_and_detect(file_path)

    def display_and_detect(self, file_path):
        if not self.detection_model:
            print("Object Detection Error: Model is not loaded.")
            return

        print(f"\nDetecting objects in: {os.path.basename(file_path)}")
        try:
            # Handle raw files
            if file_path.lower().endswith(('.nef', '.cr2', '.arw', '.orf')):
                with rawpy.imread(file_path) as raw:
                    rgb = raw.postprocess()
                    img = Image.fromarray(rgb)
            else:
                img = Image.open(file_path).convert('RGB')
            
            # Run detection on the image
            detections = self.detection_model(img)
            
            # Create a copy to draw on
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            
            bird_count = 0
            for detection in detections:
                label = detection['label']
                if 'bird' in label.lower():
                    bird_count += 1
                    box = detection['box']
                    xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                    
                    # Draw a green bounding box
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=3)
            
            # Display the image with bounding boxes
            draw_img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(draw_img)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Update the text with the number of birds
            self.predictions_text.config(text=f"Birds detected: {bird_count}")
        except Exception as e:
            print(f"Error during object detection: {e}")
            self.predictions_text.config(text=f"Error during detection: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = BirdClassifierGUI(root)
    root.mainloop()