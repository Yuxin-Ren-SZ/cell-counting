from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import tifffile
from PIL import Image, ImageTk

from core.baseline import make_overlay, segment_cells


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

        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x")

        self.threshold_var = tk.DoubleVar(value=1.0)
        self.min_size_var = tk.IntVar(value=30)
        self.min_distance_var = tk.IntVar(value=8)

        self.add_slider(control_frame, "Threshold factor", self.threshold_var, 0.5, 1.5, 0)
        self.add_slider(control_frame, "Min object size", self.min_size_var, 1, 300, 1)
        self.add_slider(control_frame, "Watershed min distance", self.min_distance_var, 1, 30, 2)

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

    def add_slider(self, parent, label, variable, min_value, max_value, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")

        slider = ttk.Scale(
            parent,
            from_=min_value,
            to=max_value,
            variable=variable,
            orient="horizontal",
            length=250,
        )
        slider.grid(row=row, column=1, padx=8)

        value_label = ttk.Label(parent, textvariable=variable)
        value_label.grid(row=row, column=2, sticky="w")

    def set_default_parameters(self):
        name = self.image_path.lower()

        if "neun" in name:
            self.threshold_var.set(0.85)
            self.min_size_var.set(45)
            self.min_distance_var.set(7)
        elif "olig2" in name:
            self.threshold_var.set(0.95)
            self.min_size_var.set(20)
            self.min_distance_var.set(10)
        else:
            self.threshold_var.set(1.0)
            self.min_size_var.set(30)
            self.min_distance_var.set(8)

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

        self.set_default_parameters()
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

        threshold_factor = self.threshold_var.get()
        min_size = self.min_size_var.get()
        min_distance = self.min_distance_var.get()

        labels, count, used_channel = segment_cells(
            image=self.image,
            image_path=self.image_path,
            threshold_factor=threshold_factor,
            min_size=min_size,
            min_distance=min_distance,
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
