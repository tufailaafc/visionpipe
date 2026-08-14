import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import imageio.v3 as iio


class ThermalNPYViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("FLIR A65 Thermal NPY Viewer")
        self.root.geometry("1400x850")

        self.folder = None
        self.files = []
        self.index = 0
        self.current_img = None
        self.current_display_img = None

        self.create_widgets()

    def create_widgets(self):
        # --------------------------------------------------
        # Top control bar
        # --------------------------------------------------
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(
            top_frame,
            text="Open Folder",
            command=self.open_folder
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            top_frame,
            text="Previous",
            command=self.previous_image
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            top_frame,
            text="Next",
            command=self.next_image
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            top_frame,
            text="Export Current PNG",
            command=lambda: self.export_current_image("png")
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            top_frame,
            text="Export Current JPG",
            command=lambda: self.export_current_image("jpg")
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            top_frame,
            text="Export All PNG",
            command=self.export_all_png
        ).pack(side=tk.LEFT, padx=3)

        self.mode_var = tk.StringVar(value="raw")

        tk.Radiobutton(
            top_frame,
            text="Raw",
            variable=self.mode_var,
            value="raw",
            command=self.refresh_image
        ).pack(side=tk.LEFT, padx=8)

        tk.Radiobutton(
            top_frame,
            text="Convert: raw × 0.04 - 273.15",
            variable=self.mode_var,
            value="celsius_004",
            command=self.refresh_image
        ).pack(side=tk.LEFT, padx=8)

        tk.Radiobutton(
            top_frame,
            text="Convert: raw × 0.01 - 273.15",
            variable=self.mode_var,
            value="celsius_001",
            command=self.refresh_image
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            top_frame,
            text="Quit / Exit",
            command=self.quit_app,
            bg="lightcoral"
        ).pack(side=tk.RIGHT, padx=5)

        # --------------------------------------------------
        # Information labels
        # --------------------------------------------------
        self.info_label = tk.Label(
            self.root,
            text="Open a folder containing .npy files"
        )
        self.info_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        self.cursor_label = tk.Label(
            self.root,
            text="Move mouse over image to read pixel value",
            font=("Arial", 10),
            bg="black",
            fg="white"
        )
        self.cursor_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        # --------------------------------------------------
        # Main body: left file list + right image viewer
        # --------------------------------------------------
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left panel
        left_frame = tk.Frame(main_frame, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Label(
            left_frame,
            text="Thermal Images",
            font=("Arial", 11, "bold")
        ).pack(side=tk.TOP, fill=tk.X)

        list_frame = tk.Frame(left_frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image_listbox = tk.Listbox(
            list_frame,
            width=50,
            height=35,
            exportselection=False
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(
            list_frame,
            orient=tk.VERTICAL,
            command=self.image_listbox.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox.config(yscrollcommand=scrollbar.set)
        self.image_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        self.file_count_label = tk.Label(
            left_frame,
            text="No folder loaded"
        )
        self.file_count_label.pack(side=tk.BOTTOM, fill=tk.X, pady=3)

        # Right panel
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = plt.Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True
        )

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    # --------------------------------------------------
    # Folder and file handling
    # --------------------------------------------------
    def open_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder containing .npy files"
        )

        if not folder:
            return

        self.folder = Path(folder)
        self.files = sorted(self.folder.glob("*.npy"))

        if not self.files:
            messagebox.showerror(
                "No files",
                "No .npy files found in this folder."
            )
            return

        self.populate_file_list()

        self.index = 0
        self.load_image()
        self.update_listbox_selection()

    def populate_file_list(self):
        self.image_listbox.delete(0, tk.END)

        for i, file_path in enumerate(self.files):
            display_name = f"{i + 1:03d}. {file_path.name}"
            self.image_listbox.insert(tk.END, display_name)

        self.file_count_label.config(
            text=f"{len(self.files)} .npy files loaded"
        )

    def on_listbox_select(self, event):
        selection = self.image_listbox.curselection()

        if not selection:
            return

        selected_index = selection[0]

        if selected_index == self.index:
            return

        self.index = selected_index
        self.load_image()

    def update_listbox_selection(self):
        if not self.files:
            return

        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.index)
        self.image_listbox.activate(self.index)
        self.image_listbox.see(self.index)

    def load_image(self):
        if not self.files:
            return

        file_path = self.files[self.index]

        try:
            img = np.load(file_path)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Could not load file:\n{file_path}\n\n{e}"
            )
            return

        img = np.squeeze(img)

        if img.ndim != 2:
            messagebox.showerror(
                "Invalid image",
                f"Expected a 2D thermal image, but got shape {img.shape}"
            )
            return

        self.current_img = img
        self.refresh_image()
        self.update_listbox_selection()

    # --------------------------------------------------
    # Thermal conversion
    # --------------------------------------------------
    def get_display_image(self):
        img = self.current_img.astype(np.float32)

        mode = self.mode_var.get()

        if mode == "celsius_004":
            img = img * 0.04 - 273.15

        elif mode == "celsius_001":
            img = img * 0.01 - 273.15

        return img

    def convert_single_value(self, raw_value):
        mode = self.mode_var.get()

        if mode == "raw":
            return raw_value, "Raw value"

        elif mode == "celsius_004":
            return raw_value * 0.04 - 273.15, "Temperature estimate (°C)"

        elif mode == "celsius_001":
            return raw_value * 0.01 - 273.15, "Temperature estimate (°C)"

        return raw_value, "Value"

    # --------------------------------------------------
    # Display image
    # --------------------------------------------------
    def refresh_image(self):
        if self.current_img is None:
            return

        self.current_display_img = self.get_display_image()
        file_path = self.files[self.index]

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        im = self.ax.imshow(
            self.current_display_img,
            cmap="inferno",
            aspect="equal"
        )

        self.ax.set_title(file_path.name, fontsize=9)
        self.ax.axis("off")

        cbar = self.fig.colorbar(
            im,
            ax=self.ax,
            fraction=0.046,
            pad=0.04
        )

        mode = self.mode_var.get()

        if mode == "raw":
            cbar.set_label("Raw value")
        else:
            cbar.set_label("Temperature estimate (°C)")

        self.canvas.draw()

        self.info_label.config(
            text=(
                f"Image {self.index + 1} of {len(self.files)} | "
                f"Shape: {self.current_img.shape} | "
                f"Dtype: {self.current_img.dtype} | "
                f"Min: {np.nanmin(self.current_img):.3f} | "
                f"Max: {np.nanmax(self.current_img):.3f} | "
                f"File: {file_path.name}"
            )
        )

        self.cursor_label.config(
            text="Move mouse over image to read pixel value"
        )

    # --------------------------------------------------
    # Mouse hover temperature reader
    # --------------------------------------------------
    def on_mouse_move(self, event):
        if self.current_img is None:
            return

        if event.inaxes != self.ax:
            self.cursor_label.config(
                text="Move mouse over image to read pixel value"
            )
            return

        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))

        height, width = self.current_img.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return

        raw_value = float(self.current_img[row, col])
        converted_value, label = self.convert_single_value(raw_value)

        self.cursor_label.config(
            text=(
                f"Mouse pixel: row={row}, col={col} | "
                f"Raw value={raw_value:.3f} | "
                f"{label}={converted_value:.3f}"
            )
        )

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------
    def next_image(self):
        if not self.files:
            return

        self.index = (self.index + 1) % len(self.files)
        self.load_image()

    def previous_image(self):
        if not self.files:
            return

        self.index = (self.index - 1) % len(self.files)
        self.load_image()

    # --------------------------------------------------
    # Export functions
    # --------------------------------------------------
    def normalize_for_export(self, img):
        """
        Convert thermal array to 8-bit RGB image using inferno colormap.
        This is for visualization only.
        It does not preserve numeric temperature values.
        """

        img = img.astype(np.float32)

        min_val = np.nanmin(img)
        max_val = np.nanmax(img)

        if max_val == min_val:
            normalized = np.zeros_like(img, dtype=np.float32)
        else:
            normalized = (img - min_val) / (max_val - min_val)

        colored = cm.inferno(normalized)

        rgb = (colored[:, :, :3] * 255).astype(np.uint8)

        return rgb

    def export_current_image(self, file_format="png"):
        if self.current_img is None:
            messagebox.showerror(
                "No image",
                "No image is currently loaded."
            )
            return

        img_display = self.get_display_image()
        rgb = self.normalize_for_export(img_display)

        current_name = self.files[self.index].stem
        default_name = f"{current_name}.{file_format}"

        if file_format.lower() in ["jpg", "jpeg"]:
            filetypes = [
                ("JPEG files", "*.jpg"),
                ("JPEG files", "*.jpeg"),
                ("All files", "*.*")
            ]
            defaultextension = ".jpg"
        else:
            filetypes = [
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
            defaultextension = ".png"

        save_path = filedialog.asksaveasfilename(
            title=f"Save current image as {file_format.upper()}",
            defaultextension=defaultextension,
            initialfile=default_name,
            filetypes=filetypes
        )

        if not save_path:
            return

        try:
            iio.imwrite(save_path, rgb)
            messagebox.showinfo(
                "Export complete",
                f"Saved image:\n{save_path}"
            )

        except Exception as e:
            messagebox.showerror(
                "Export error",
                f"Could not save image:\n{save_path}\n\n{e}"
            )

    def export_all_png(self):
        if not self.files:
            messagebox.showerror(
                "No folder",
                "No image folder is loaded."
            )
            return

        output_folder = filedialog.askdirectory(
            title="Select output folder for PNG images"
        )

        if not output_folder:
            return

        output_folder = Path(output_folder)
        mode = self.mode_var.get()

        exported_count = 0
        skipped_count = 0

        try:
            for file_path in self.files:
                img = np.load(file_path)
                img = np.squeeze(img)

                if img.ndim != 2:
                    skipped_count += 1
                    continue

                img = img.astype(np.float32)

                if mode == "celsius_004":
                    img = img * 0.04 - 273.15

                elif mode == "celsius_001":
                    img = img * 0.01 - 273.15

                rgb = self.normalize_for_export(img)

                save_path = output_folder / f"{file_path.stem}.png"
                iio.imwrite(save_path, rgb)

                exported_count += 1

            messagebox.showinfo(
                "Export complete",
                (
                    f"Exported {exported_count} PNG images to:\n"
                    f"{output_folder}\n\n"
                    f"Skipped files: {skipped_count}"
                )
            )

        except Exception as e:
            messagebox.showerror(
                "Export error",
                f"Could not export all images.\n\n{e}"
            )

    # --------------------------------------------------
    # Quit app
    # --------------------------------------------------
    def quit_app(self):
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalNPYViewer(root)
    root.mainloop()
