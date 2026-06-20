from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import tifffile
from PIL import Image, ImageTk

from segmentation import make_overlay, segment_cells

def numpy_to_tk_image(image_array, max_width=900, max_height=550):
    image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image_array)
    pil_image.thumbnail((max_width, max_height))

    return ImageTk.PhotoImage(pil_image)

class CellCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cell Counting")

        self.image_path = None
        self.image = None
        self.tk_preview = None

        self.build_ui()

    def build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        choose_button = ttk.Button(
            top_frame,
            text="Choose TIFF Image",
            command=self.choose_file,
        )
        choose_button.pack(side="left")

        self.file_label = ttk.Label(top_frame, text="No file selected")
        self.file_label.pack(side="left", padx=10)

        run_button = ttk.Button(
            self.root,
            text="Run Segmentation",
            command=self.run_segmentation,
        )
        run_button.pack(pady=10)


        self.result_label = ttk.Label(
            self.root,
            text="Cell count: -",
            font=("Arial", 18, "bold"),
            padding=10,
        )
        self.result_label.pack(fill="x")

        self.preview_label = ttk.Label(self.root)
        self.preview_label.pack(padx=10, pady=10)

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Choose TIFF image",
            filetypes=[
                ("TIFF images", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return
        
        self.image_path = path
        self.image = tifffile.imread(path)
        self.file_label.config(text=Path(path).name)

        self.show_original_image()

    def show_original_image(self):
        display = self.image[:, :, :3].astype(float)

        if display.max() > 1:
            display = display / 255.0

        self.tk_preview = numpy_to_tk_image(display)
        self.preview_label.config(image=self.tk_preview)

    def run_segmentation(self):
        if self.image is None:
            self.result_label.config(text="Please choose a TIFF image first.")
            return

        labels, count, used_channel = segment_cells(
            image=self.image,
            image_path=self.image_path,
        )

        display, overlay = make_overlay(self.image, labels)

        output_path = Path(self.image_path).with_name(
            Path(self.image_path).stem + "_tk_overlay.png"
        )

        Image.fromarray((overlay * 255).astype(np.uint8)).save(output_path)

        self.tk_preview = numpy_to_tk_image(overlay)
        self.preview_label.config(image=self.tk_preview)

        self.result_label.config(
            text=f"Cell count: {count} | Detected channel: {used_channel}"
        )
if __name__ == "__main__":
    root = tk.Tk()
    app = CellCounterApp(root)
    root.mainloop()