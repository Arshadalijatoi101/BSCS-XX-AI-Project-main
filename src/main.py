import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import AIEducationApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = AIEducationApp(root)
    root.mainloop()