# GUI.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import os
import sys
import threading
from io import StringIO
from ML import BirdClassifierModel, ObjectDetectorModel

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

class BirdClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("Bird Photo Classifier")
        master.geometry("800x600")

        self.classifier_model = BirdClassifierModel()
        self.object_detector_model = ObjectDetectorModel()
        self.console_window = None

        self.create_menu()
        self.create_widgets()
        
        # Load model and print to buffer
        self.load_models()
        
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
            label=f"Classification Model: {self.classifier_model.model_info['name']}",
            command=self.show_model_info_classification
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
            
            self.redirector = ConsoleRedirector(console_text, self.classifier_model.console_buffer.getvalue())

    def open_object_detection_gui(self):
        detection_window = tk.Toplevel(self.master)
        ObjectDetectionWindow(detection_window, self.object_detector_model)

    def show_model_info_classification(self):
        messagebox.showinfo("Model Information", self.classifier_model.model_info['description'])

    def load_models(self):
        self.classifier_model.load_model()
        self.object_detector_model.load_model()

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.nef *.cr2 *.arw *.orf")]
        )
        if file_path:
            self.display_image(file_path)
            self.classify_image(file_path)

    def display_image(self, file_path):
        try:
            img = self.classifier_model.load_image_from_path(file_path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.image_label.config(text=f"Error loading image: {e}")
            self.image_label.image = None
            
    def classify_image(self, file_path):
        try:
            input_image = self.classifier_model.load_image_from_path(file_path)
            predictions = self.classifier_model.classify(input_image)
            
            if predictions:
                predictions_str = ""
                for i in range(min(5, len(predictions))):
                    pred = predictions[i]
                    scientific_name = pred['label']
                    common_name = self.classifier_model.get_common_name(scientific_name)
                    confidence = pred['score']
                    predictions_str += f"{i+1}. {common_name}\n   Confidence: {confidence:.4f}\n"
                self.predictions_text.config(text=predictions_str)
            else:
                self.predictions_text.config(text="Could not classify the image.")
        except Exception as e:
            print(f"Error classifying image: {e}")

class ObjectDetectionWindow:
    def __init__(self, master, model):
        self.master = master
        self.model = model
        master.title("Object Detection")
        master.geometry("800x600")

        self.create_widgets()

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

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.nef *.cr2 *.arw *.orf")]
        )
        if file_path:
            self.display_and_detect(file_path)

    def display_and_detect(self, file_path):
        if not self.model.is_loaded():
            print("Object Detection Error: Model is not loaded.")
            return

        print(f"\nDetecting objects in: {os.path.basename(file_path)}")
        try:
            img = self.model.load_image_from_path(file_path)
            detections = self.model.detect(img)
            
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)
            
            bird_count = 0
            for detection in detections:
                label = detection['label']
                if 'bird' in label.lower():
                    bird_count += 1
                    box = detection['box']
                    xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                    
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=3)
            
            draw_img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(draw_img)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            self.predictions_text.config(text=f"Birds detected: {bird_count}")
        except Exception as e:
            print(f"Error during object detection: {e}")
            self.predictions_text.config(text=f"Error during detection: {e}")

