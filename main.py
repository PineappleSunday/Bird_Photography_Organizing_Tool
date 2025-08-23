# Main.py

import tkinter as tk
from GUI import BirdClassifierGUI

if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    
    # Initialize the GUI
    gui = BirdClassifierGUI(root)
    
    # Start the Tkinter event loop
    root.mainloop()
