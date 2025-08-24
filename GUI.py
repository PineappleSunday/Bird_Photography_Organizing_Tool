# GUI.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import os
import sys
import threading
from io import StringIO
from ML import BirdClassifierModel, ObjectDetectorModel
import rawpy

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
        master.title("Bird Photo Classifier and Detector")
        # Open the image using Pillow
        image = Image.open("Bird_Classifier_Icon.png")
        # Convert the image to a Tkinter-compatible format
        photo = ImageTk.PhotoImage(image)

        # Set the window icon using the PhotoImage object
        master.iconphoto(False, photo) 
        master.geometry("1000x800")

        self.classifier_model = BirdClassifierModel()
        self.object_detector_model = ObjectDetectorModel()
        self.console_window = None
        self.original_image = None
        self.zoom_level = 1.0
        self.current_file_path = None
        self.last_predictions = None

        self.create_menu()
        self.create_widgets()
        
        console_buffer = StringIO()
        sys.stdout = console_buffer
        self.load_models()
        self.initial_output = console_buffer.getvalue()
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
        
    def create_widgets(self):
        self.image_label = tk.Label(self.master)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)
        self.image_label.bind("<MouseWheel>", self.on_mouse_wheel)

        self.info_frame = tk.Frame(self.master)
        self.info_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        # Classification Predictions Frame
        self.predictions_frame = tk.Frame(self.info_frame)
        self.predictions_frame.pack(pady=(0, 20), fill=tk.X)
        self.title_label = tk.Label(self.predictions_frame, text="Top 5 Predictions", font=("Helvetica", 16))
        self.title_label.pack()
        self.predictions_text = tk.Label(self.predictions_frame, justify=tk.LEFT, font=("Helvetica", 12))
        self.predictions_text.pack()
        
        # Object Detection Frame
        self.detection_frame = tk.Frame(self.info_frame)
        self.detection_frame.pack(pady=(20, 0), fill=tk.X)
        self.detection_title = tk.Label(self.detection_frame, text="Detected Birds", font=("Helvetica", 16))
        self.detection_title.pack()
        self.detection_text = tk.Label(self.detection_frame, justify=tk.LEFT, font=("Helvetica", 12))
        self.detection_text.pack()

        self.open_button = tk.Button(self.info_frame, text="Open Image", command=self.open_file)
        self.open_button.pack(pady=(20, 0))

        self.rename_button = tk.Button(self.info_frame, text="Rename Image", command=self.rename_file)
        self.rename_button.pack(pady=(10, 0))

    def on_mouse_wheel(self, event):
        if self.original_image:
            if event.delta > 0:
                self.zoom_level *= 1.1  # Zoom in
            else:
                self.zoom_level /= 1.1  # Zoom out
            
            self.display_zoomed_image()
            
    def display_zoomed_image(self):
        if self.original_image:
            width, height = self.original_image.size
            new_width = int(width * self.zoom_level)
            new_height = int(height * self.zoom_level)
            
            # Ensure the image does not exceed the window size
            max_width = self.master.winfo_width() - self.info_frame.winfo_width() - 20
            max_height = self.master.winfo_height() - 20
            
            new_width = min(new_width, max_width)
            new_height = min(new_height, max_height)

            if new_width > 0 and new_height > 0:
                resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(resized_image)
                self.image_label.config(image=photo)
                self.image_label.image = photo

    def open_console(self):
        if self.console_window is None or not self.console_window.winfo_exists():
            self.console_window = tk.Toplevel(self.master)
            self.console_window.title("Console Output")
            self.console_window.geometry("600x400")
            
            console_text = scrolledtext.ScrolledText(self.console_window, wrap=tk.WORD, state='normal')
            console_text.pack(expand=True, fill='both')
            
            self.redirector = ConsoleRedirector(console_text, self.initial_output)

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
            self.current_file_path = file_path
            self.process_image(file_path)

    def process_image(self, file_path):
        try:
            # Load the image just once
            base_image = self.classifier_model.load_image_from_path(file_path)
            
            # Run object detection first
            if not self.object_detector_model.is_loaded():
                print("Object Detection Error: Model is not loaded.")
                return

            print(f"\nDetecting objects in: {os.path.basename(file_path)}")
            detections = self.object_detector_model.detect(base_image)
            
            draw_img = base_image.copy()
            draw = ImageDraw.Draw(draw_img)
            
            bird_count = 0
            for detection in detections:
                label = detection['label']
                if 'bird' in label.lower() and detection['score'] > 0.90:  # Add confidence check
                    bird_count += 1
                    box = detection['box']
                    xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=10)
            
            self.original_image = draw_img
            self.detection_text.config(text=f"Birds detected: {bird_count}")

            # Then run classification on the same base image
            if not self.classifier_model.is_loaded():
                print("Classification Error: Model is not loaded.")
                return

            print(f"Classifying: {os.path.basename(file_path)}")
            self.last_predictions = self.classifier_model.classify(base_image)
            
            if self.last_predictions:
                predictions_str = ""
                for i in range(min(5, len(self.last_predictions))):
                    pred = self.last_predictions[i]
                    scientific_name = pred['label']
                    common_name = self.classifier_model.get_common_name(scientific_name)
                    confidence = pred['score']
                    predictions_str += f"{i+1}. {common_name}\n   Confidence: {confidence:.4f}\n"
                self.predictions_text.config(text=predictions_str)
            else:
                self.predictions_text.config(text="Could not classify the image.")
            
            # Finally, display the processed image
            display_width = 500
            display_height = 500
            img_width, img_height = self.original_image.size
            self.zoom_level = min(display_width / img_width, display_height / img_height, 1.0)
            self.display_zoomed_image()
            
        except Exception as e:
            print(f"Error during image processing: {e}")
            self.image_label.config(text=f"Error loading image: {e}")
            self.predictions_text.config(text="")
            self.detection_text.config(text="")
            self.image_label.image = None
            self.original_image = None
            self.current_file_path = None
    
    def rename_file(self):
        if not self.current_file_path:
            messagebox.showerror("Error", "No image is open to rename.")
            return

        if not self.last_predictions:
            messagebox.showerror("Error", "No predictions available to rename the file.")
            return

        try:
            # Get the top prediction's common name
            top_prediction = self.last_predictions[0]
            scientific_name = top_prediction['label']
            common_name = self.classifier_model.get_common_name(scientific_name)

            # Sanitize the name for use as a filename
            base_name = common_name.replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")
            
            # Get the original file extension
            root, extension = os.path.splitext(self.current_file_path)
            
            # Construct the new path
            directory = os.path.dirname(self.current_file_path)
            new_file_path = os.path.join(directory, f"{base_name}{extension}")

            # Rename the file
            os.rename(self.current_file_path, new_file_path)
            self.current_file_path = new_file_path
            messagebox.showinfo("Success", f"File renamed to:\n{os.path.basename(new_file_path)}")
            print(f"File renamed to: {new_file_path}")
        except Exception as e:
            messagebox.showerror("Rename Error", f"An error occurred while renaming the file: {e}")
            print(f"Error renaming file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = BirdClassifierGUI(root)
    root.mainloop()
