import re
import csv
import shutil
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image


class RGBThermalSyncViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB + Thermal Synchronization Viewer with Tagging")
        self.root.geometry("1780x1000")

        self.rgb_folder = None
        self.thermal_folder = None

        self.rgb_records_all = []
        self.thermal_records_all = []
        self.rgb_records_filtered = []

        self.rgb_index = 0
        self.thermal_candidates = []
        self.selected_thermal_index = 0

        self.current_rgb_img = None
        self.current_thermal_img = None

        # Rotation angles for visual inspection only
        self.rgb_rotation_angle = 0
        self.thermal_rotation_angle = 0

        # Tagged pairs
        # key = rgb_path string
        # value = dictionary containing RGB record, thermal record, delta, rotations
        self.tagged_pairs = {}

        self.create_widgets()

    # --------------------------------------------------
    # GUI
    # --------------------------------------------------
    def create_widgets(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(
            top_frame,
            text="Open RGB Folder",
            command=self.open_rgb_folder,
            bg="lightgreen"
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Open Thermal Folder",
            command=self.open_thermal_folder,
            bg="orange"
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Previous RGB",
            command=self.previous_rgb
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Next RGB",
            command=self.next_rgb
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="RGB ⟲",
            command=lambda: self.rotate_rgb(-90)
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="RGB ⟳",
            command=lambda: self.rotate_rgb(90)
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Thermal ⟲",
            command=lambda: self.rotate_thermal(-90)
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Thermal ⟳",
            command=lambda: self.rotate_thermal(90)
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Reset Rotation",
            command=self.reset_rotation
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Tag Current Pair",
            command=self.tag_current_pair,
            bg="yellow"
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Untag Current RGB",
            command=self.untag_current_rgb
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Export Tagged Pairs",
            command=self.export_tagged_pairs,
            bg="lightblue"
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Help",
            command=self.show_help,
            bg="lightcyan"
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            top_frame,
            text="Quit / Exit",
            command=self.quit_app,
            bg="lightcoral"
        ).pack(side=tk.RIGHT, padx=5)

        # --------------------------------------------------
        # Query area
        # --------------------------------------------------
        query_frame = tk.LabelFrame(self.root, text="Query RGB Images")
        query_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(query_frame, text="RFID contains:").grid(
            row=0, column=0, padx=5, pady=3, sticky="e"
        )

        self.rfid_entry = tk.Entry(query_frame, width=30)
        self.rfid_entry.grid(row=0, column=1, padx=5, pady=3, sticky="w")

        tk.Label(
            query_frame,
            text="Example: 406114884 or LA CAN 000406114884",
            fg="gray"
        ).grid(row=1, column=1, padx=5, sticky="w")

        tk.Label(query_frame, text="Single Date:").grid(
            row=0, column=2, padx=5, pady=3, sticky="e"
        )

        self.date_entry = tk.Entry(query_frame, width=15)
        self.date_entry.grid(row=0, column=3, padx=5, pady=3, sticky="w")

        tk.Label(
            query_frame,
            text="YYYYMMDD, e.g., 20260425",
            fg="gray"
        ).grid(row=1, column=3, padx=5, sticky="w")

        tk.Label(query_frame, text="Start Date/Time:").grid(
            row=0, column=4, padx=5, pady=3, sticky="e"
        )

        self.start_entry = tk.Entry(query_frame, width=22)
        self.start_entry.grid(row=0, column=5, padx=5, pady=3, sticky="w")

        tk.Label(
            query_frame,
            text="Example: 20260425 000000",
            fg="gray"
        ).grid(row=1, column=5, padx=5, sticky="w")

        tk.Label(query_frame, text="End Date/Time:").grid(
            row=0, column=6, padx=5, pady=3, sticky="e"
        )

        self.end_entry = tk.Entry(query_frame, width=22)
        self.end_entry.grid(row=0, column=7, padx=5, pady=3, sticky="w")

        tk.Label(
            query_frame,
            text="Example: 20260428 235959",
            fg="gray"
        ).grid(row=1, column=7, padx=5, sticky="w")

        tk.Label(query_frame, text="Thermal tolerance ± seconds:").grid(
            row=0, column=8, padx=5, pady=3, sticky="e"
        )

        self.tolerance_entry = tk.Entry(query_frame, width=8)
        self.tolerance_entry.insert(0, "5")
        self.tolerance_entry.grid(row=0, column=9, padx=5, pady=3, sticky="w")

        tk.Label(
            query_frame,
            text="Try 5, 10, 30, 60",
            fg="gray"
        ).grid(row=1, column=9, padx=5, sticky="w")

        self.same_rfid_var = tk.BooleanVar(value=True)

        tk.Checkbutton(
            query_frame,
            text="Match same RFID only",
            variable=self.same_rfid_var,
            command=self.reload_current_rgb
        ).grid(row=0, column=10, padx=5, pady=3, sticky="w")

        tk.Label(
            query_frame,
            text="Uncheck only for camera-time debugging",
            fg="gray"
        ).grid(row=1, column=10, padx=5, sticky="w")

        tk.Button(
            query_frame,
            text="Apply Query",
            command=self.apply_query,
            bg="lightgreen"
        ).grid(row=0, column=11, padx=5, pady=3)

        tk.Button(
            query_frame,
            text="Clear Query",
            command=self.clear_query
        ).grid(row=0, column=12, padx=5, pady=3)

        tk.Label(
            query_frame,
            text=(
                "Tip: Tag Current Pair saves the selected RGB image and selected thermal candidate as a good dataset pair. "
                "Export Tagged Pairs copies all tagged RGB/thermal originals and writes a manifest CSV."
            ),
            fg="blue"
        ).grid(row=2, column=0, columnspan=13, padx=5, pady=5, sticky="w")

        # --------------------------------------------------
        # Info labels
        # --------------------------------------------------
        self.folder_info_label = tk.Label(
            self.root,
            text="Open RGB and Thermal folders to begin."
        )
        self.folder_info_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        self.sync_info_label = tk.Label(
            self.root,
            text="Synchronization information will appear here.",
            font=("Arial", 10),
            bg="black",
            fg="white"
        )
        self.sync_info_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        self.tag_info_label = tk.Label(
            self.root,
            text="Tagged pairs: 0",
            font=("Arial", 10, "bold"),
            bg="darkgreen",
            fg="white"
        )
        self.tag_info_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        # --------------------------------------------------
        # Main frame
        # --------------------------------------------------
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_frame, width=430)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Label(
            left_panel,
            text="Filtered RGB Images",
            font=("Arial", 11, "bold")
        ).pack(side=tk.TOP, fill=tk.X)

        rgb_list_frame = tk.Frame(left_panel)
        rgb_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.rgb_listbox = tk.Listbox(
            rgb_list_frame,
            width=75,
            height=38,
            exportselection=False
        )
        self.rgb_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        rgb_scrollbar = tk.Scrollbar(
            rgb_list_frame,
            orient=tk.VERTICAL,
            command=self.rgb_listbox.yview
        )
        rgb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.rgb_listbox.config(yscrollcommand=rgb_scrollbar.set)
        self.rgb_listbox.bind("<<ListboxSelect>>", self.on_rgb_select)

        self.rgb_count_label = tk.Label(
            left_panel,
            text="No RGB folder loaded"
        )
        self.rgb_count_label.pack(side=tk.BOTTOM, fill=tk.X, pady=3)

        display_panel = tk.Frame(main_frame)
        display_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        figures_frame = tk.Frame(display_panel)
        figures_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        rgb_fig_frame = tk.LabelFrame(figures_frame, text="RGB Image")
        rgb_fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.rgb_fig = plt.Figure(figsize=(6, 5))
        self.rgb_ax = self.rgb_fig.add_subplot(111)
        self.rgb_canvas = FigureCanvasTkAgg(self.rgb_fig, master=rgb_fig_frame)
        self.rgb_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        thermal_fig_frame = tk.LabelFrame(figures_frame, text="Thermal Image")
        thermal_fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.thermal_fig = plt.Figure(figsize=(6, 5))
        self.thermal_ax = self.thermal_fig.add_subplot(111)
        self.thermal_canvas = FigureCanvasTkAgg(self.thermal_fig, master=thermal_fig_frame)
        self.thermal_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        candidates_frame = tk.LabelFrame(
            display_panel,
            text="Thermal Candidates for Selected RGB Image"
        )
        candidates_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.thermal_listbox = tk.Listbox(
            candidates_frame,
            width=180,
            height=9,
            exportselection=False
        )
        self.thermal_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        thermal_scrollbar = tk.Scrollbar(
            candidates_frame,
            orient=tk.VERTICAL,
            command=self.thermal_listbox.yview
        )
        thermal_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.thermal_listbox.config(yscrollcommand=thermal_scrollbar.set)
        self.thermal_listbox.bind("<<ListboxSelect>>", self.on_thermal_select)

    # --------------------------------------------------
    # Help
    # --------------------------------------------------
    def show_help(self):
        help_text = """
RGB + THERMAL SYNCHRONIZATION VIEWER WITH TAGGING

Purpose:
This tool helps you visually confirm RGB and thermal pairs for the same pig/RFID
at nearly the same time.

Workflow:
1. Open RGB Folder.
2. Open Thermal Folder.
3. Query RFID/date/time if needed.
4. Select RGB image from the left list.
5. Review thermal candidates at the bottom.
6. Click the best thermal candidate.
7. Rotate RGB or thermal if needed for visual inspection.
8. Click Tag Current Pair.
9. Repeat for good pairs.
10. Click Export Tagged Pairs.

Export creates:
selected_output_folder/
    RGB/
    Thermal/
    manifest.csv

The exported images are copied from the original files.
Rotation is saved only in manifest.csv as metadata; it does not alter original images.

Filename parsing:
RGB may look like:
LA CAN 000406114884_409122300405_color_SIDE_20260421_210016_588897.png

Thermal may look like:
LA CAN 000406114884_flir_raw_20260421_210016_849846.npy

The program extracts RFID as everything before the first underscore:
LA CAN 000406114884

The timestamp is extracted from the last three fields:
YYYYMMDD_HHMMSS_fraction
"""
        messagebox.showinfo("Help / Examples", help_text)

    # --------------------------------------------------
    # Filename parser
    # --------------------------------------------------
    def parse_filename(self, file_path):
        stem = file_path.stem

        pattern = re.compile(
            r"^(?P<prefix>.+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<fraction>\d+)$"
        )

        match = pattern.match(stem)

        if not match:
            return None

        data = match.groupdict()

        prefix = data["prefix"]
        parts = prefix.split("_")

        animal_rfid = parts[0].strip()
        metadata_parts = parts[1:]

        camera = metadata_parts[0] if len(metadata_parts) >= 1 else ""
        mode = "_".join(metadata_parts[1:]) if len(metadata_parts) >= 2 else ""

        try:
            dt = datetime.strptime(data["date"] + data["time"], "%Y%m%d%H%M%S")
        except ValueError:
            return None

        fraction = data["fraction"]

        if fraction:
            microsecond_string = fraction[:6].ljust(6, "0")
            microsecond = int(microsecond_string)
            dt = dt.replace(microsecond=microsecond)

        return {
            "path": file_path,
            "filename": file_path.name,
            "stem": stem,
            "rfid": animal_rfid,
            "rfid_lower": animal_rfid.lower(),
            "prefix": prefix,
            "metadata": "_".join(metadata_parts),
            "camera": camera,
            "mode": mode,
            "date": data["date"],
            "time": data["time"],
            "fraction": fraction,
            "datetime": dt,
            "extension": file_path.suffix.lower()
        }

    def parse_query_datetime(self, text):
        text = text.strip()

        if not text:
            return None

        text = text.replace("_", " ")
        text = " ".join(text.split())

        formats = [
            "%Y%m%d %H%M%S",
            "%Y%m%d%H%M%S"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass

        raise ValueError(
            f"Invalid datetime format: {text}\n\n"
            "Use one of these formats:\n"
            "20260425 000000\n"
            "20260425_000000\n"
            "20260425000000"
        )

    # --------------------------------------------------
    # Folder loading
    # --------------------------------------------------
    def open_rgb_folder(self):
        folder = filedialog.askdirectory(title="Select RGB image folder")

        if not folder:
            return

        self.rgb_folder = Path(folder)

        files = self.collect_files(self.rgb_folder, [".png", ".jpg", ".jpeg"])
        self.rgb_records_all, skipped = self.parse_records_from_files(files)

        if not self.rgb_records_all:
            messagebox.showerror(
                "No valid RGB files",
                "No RGB files matched the expected filename pattern."
            )
            return

        self.rgb_records_all.sort(key=lambda x: x["datetime"])
        self.rgb_records_filtered = list(self.rgb_records_all)
        self.rgb_index = 0

        self.populate_rgb_list()
        self.load_selected_rgb_and_find_thermal()
        self.update_folder_info()

        self.rgb_count_label.config(
            text=(
                f"{len(self.rgb_records_filtered)} shown | "
                f"{len(self.rgb_records_all)} valid RGB | "
                f"{skipped} skipped"
            )
        )

    def open_thermal_folder(self):
        folder = filedialog.askdirectory(title="Select thermal image folder")

        if not folder:
            return

        self.thermal_folder = Path(folder)

        files = self.collect_files(
            self.thermal_folder,
            [".png", ".jpg", ".jpeg", ".npy"]
        )

        self.thermal_records_all, skipped = self.parse_records_from_files(files)

        if not self.thermal_records_all:
            messagebox.showerror(
                "No valid thermal files",
                "No thermal files matched the expected filename pattern."
            )
            return

        self.thermal_records_all.sort(key=lambda x: x["datetime"])
        self.update_folder_info()

        if self.rgb_records_filtered:
            self.load_selected_rgb_and_find_thermal()

        messagebox.showinfo(
            "Thermal folder loaded",
            (
                f"Loaded {len(self.thermal_records_all)} valid thermal files.\n"
                f"Skipped {skipped} files."
            )
        )

    def collect_files(self, folder, extensions):
        files = []

        for ext in extensions:
            files.extend(folder.glob(f"*{ext}"))
            files.extend(folder.glob(f"*{ext.upper()}"))

        return sorted(list(set(files)))

    def parse_records_from_files(self, files):
        records = []
        skipped = 0

        for file_path in files:
            record = self.parse_filename(file_path)

            if record is None:
                skipped += 1
                continue

            records.append(record)

        return records, skipped

    def update_folder_info(self):
        rgb_text = "RGB: not loaded"
        thermal_text = "Thermal: not loaded"

        if self.rgb_folder:
            rgb_text = f"RGB: {self.rgb_folder} | {len(self.rgb_records_all)} valid files"

        if self.thermal_folder:
            thermal_text = f"Thermal: {self.thermal_folder} | {len(self.thermal_records_all)} valid files"

        self.folder_info_label.config(text=f"{rgb_text}     ||     {thermal_text}")

    # --------------------------------------------------
    # Query
    # --------------------------------------------------
    def apply_query(self):
        if not self.rgb_records_all:
            messagebox.showerror("No RGB folder", "Please open an RGB folder first.")
            return

        rfid_query = self.rfid_entry.get().strip().lower()
        date_query = self.date_entry.get().strip()

        try:
            start_dt = self.parse_query_datetime(self.start_entry.get())
            end_dt = self.parse_query_datetime(self.end_entry.get())
        except ValueError as e:
            messagebox.showerror("Invalid date/time", str(e))
            return

        if date_query:
            if not re.fullmatch(r"\d{8}", date_query):
                messagebox.showerror(
                    "Invalid date",
                    "Single Date must be in YYYYMMDD format.\n\nExample: 20260425"
                )
                return

        if start_dt and end_dt and start_dt > end_dt:
            messagebox.showerror(
                "Invalid range",
                "Start Date/Time must be earlier than End Date/Time."
            )
            return

        results = []

        for record in self.rgb_records_all:
            keep = True

            if rfid_query and rfid_query not in record["rfid_lower"]:
                keep = False

            if date_query and record["date"] != date_query:
                keep = False

            if start_dt and record["datetime"] < start_dt:
                keep = False

            if end_dt and record["datetime"] > end_dt:
                keep = False

            if keep:
                results.append(record)

        self.rgb_records_filtered = results
        self.rgb_index = 0

        self.populate_rgb_list()

        if self.rgb_records_filtered:
            self.load_selected_rgb_and_find_thermal()
            self.update_rgb_selection()
        else:
            self.clear_images()
            self.sync_info_label.config(text="No RGB images matched the query.")

        self.rgb_count_label.config(
            text=f"{len(self.rgb_records_filtered)} shown | {len(self.rgb_records_all)} total valid RGB"
        )

    def clear_query(self):
        self.rfid_entry.delete(0, tk.END)
        self.date_entry.delete(0, tk.END)
        self.start_entry.delete(0, tk.END)
        self.end_entry.delete(0, tk.END)

        self.rgb_records_filtered = list(self.rgb_records_all)
        self.rgb_index = 0

        self.populate_rgb_list()

        if self.rgb_records_filtered:
            self.load_selected_rgb_and_find_thermal()
            self.update_rgb_selection()
        else:
            self.clear_images()

        self.rgb_count_label.config(
            text=f"{len(self.rgb_records_filtered)} shown | {len(self.rgb_records_all)} total valid RGB"
        )

    def reload_current_rgb(self):
        if self.rgb_records_filtered:
            self.load_selected_rgb_and_find_thermal()

    def get_tolerance_seconds(self):
        text = self.tolerance_entry.get().strip()

        if not text:
            return 5.0

        try:
            value = float(text)
        except ValueError:
            messagebox.showerror(
                "Invalid tolerance",
                "Thermal tolerance must be a number of seconds, for example 5 or 10."
            )
            return None

        if value < 0:
            messagebox.showerror(
                "Invalid tolerance",
                "Thermal tolerance must be zero or positive."
            )
            return None

        return value

    # --------------------------------------------------
    # Listbox handling
    # --------------------------------------------------
    def populate_rgb_list(self):
        self.rgb_listbox.delete(0, tk.END)

        for i, record in enumerate(self.rgb_records_filtered):
            tag_mark = "★ " if self.is_rgb_tagged(record) else ""

            display_name = (
                f"{tag_mark}{i + 1:04d}. "
                f"RFID={record['rfid']} | "
                f"{record['date']} {record['time']}.{record['fraction']} | "
                f"meta={record['metadata']} | "
                f"{record['filename']}"
            )
            self.rgb_listbox.insert(tk.END, display_name)

    def update_rgb_selection(self):
        if not self.rgb_records_filtered:
            return

        self.rgb_listbox.selection_clear(0, tk.END)
        self.rgb_listbox.selection_set(self.rgb_index)
        self.rgb_listbox.activate(self.rgb_index)
        self.rgb_listbox.see(self.rgb_index)

    def populate_thermal_candidates_list(self):
        self.thermal_listbox.delete(0, tk.END)

        if not self.thermal_records_all:
            self.thermal_listbox.insert(
                tk.END,
                "Thermal folder is not loaded. Click Open Thermal Folder first."
            )
            return

        if not self.thermal_candidates:
            self.thermal_listbox.insert(
                tk.END,
                "No thermal candidates found. Try increasing tolerance or uncheck 'Match same RFID only'."
            )
            return

        for i, item in enumerate(self.thermal_candidates):
            record = item["record"]
            delta = item["delta_seconds"]
            within = item["within_tolerance"]

            tag = "WITHIN" if within else "NEAREST"

            display_name = (
                f"{i + 1:03d}. {tag} | "
                f"Δt={delta:+.3f} sec | "
                f"RFID={record['rfid']} | "
                f"{record['date']} {record['time']}.{record['fraction']} | "
                f"meta={record['metadata']} | "
                f"{record['filename']}"
            )
            self.thermal_listbox.insert(tk.END, display_name)

        self.thermal_listbox.selection_clear(0, tk.END)
        self.thermal_listbox.selection_set(self.selected_thermal_index)
        self.thermal_listbox.activate(self.selected_thermal_index)
        self.thermal_listbox.see(self.selected_thermal_index)

    def on_rgb_select(self, event):
        selection = self.rgb_listbox.curselection()

        if not selection:
            return

        selected_index = selection[0]

        if selected_index == self.rgb_index:
            return

        self.rgb_index = selected_index
        self.load_selected_rgb_and_find_thermal()

    def on_thermal_select(self, event):
        selection = self.thermal_listbox.curselection()

        if not selection or not self.thermal_candidates:
            return

        selected_index = selection[0]

        if selected_index >= len(self.thermal_candidates):
            return

        self.selected_thermal_index = selected_index
        self.load_selected_thermal()

    # --------------------------------------------------
    # Synchronization logic
    # --------------------------------------------------
    def find_thermal_candidates_for_rgb(self, rgb_record):
        tolerance = self.get_tolerance_seconds()

        if tolerance is None:
            return []

        if not self.thermal_records_all:
            return []

        same_rfid_only = self.same_rfid_var.get()

        candidate_pool = []

        for thermal_record in self.thermal_records_all:
            if same_rfid_only:
                if thermal_record["rfid_lower"] != rgb_record["rfid_lower"]:
                    continue

            delta_seconds = (
                thermal_record["datetime"] - rgb_record["datetime"]
            ).total_seconds()

            candidate_pool.append({
                "record": thermal_record,
                "delta_seconds": delta_seconds,
                "abs_delta_seconds": abs(delta_seconds),
                "within_tolerance": abs(delta_seconds) <= tolerance
            })

        candidate_pool.sort(key=lambda x: x["abs_delta_seconds"])

        within = [x for x in candidate_pool if x["within_tolerance"]]

        if within:
            return within[:50]

        return candidate_pool[:10]

    def load_selected_rgb_and_find_thermal(self):
        if not self.rgb_records_filtered:
            return

        rgb_record = self.rgb_records_filtered[self.rgb_index]

        try:
            self.current_rgb_img = self.load_image_file(rgb_record["path"])
        except Exception as e:
            messagebox.showerror(
                "RGB load error",
                f"Could not load RGB image:\n{rgb_record['path']}\n\n{e}"
            )
            return

        self.display_rgb_image()

        self.thermal_candidates = self.find_thermal_candidates_for_rgb(rgb_record)
        self.selected_thermal_index = 0

        self.populate_thermal_candidates_list()

        if self.thermal_candidates:
            self.load_selected_thermal()
        else:
            self.current_thermal_img = None
            self.display_no_thermal_image()
            self.update_sync_info_no_match()

        self.update_rgb_selection()
        self.update_tag_info()

    def load_selected_thermal(self):
        if not self.thermal_candidates:
            return

        rgb_record = self.rgb_records_filtered[self.rgb_index]
        item = self.thermal_candidates[self.selected_thermal_index]
        thermal_record = item["record"]

        try:
            self.current_thermal_img = self.load_image_file(thermal_record["path"])
        except Exception as e:
            messagebox.showerror(
                "Thermal load error",
                f"Could not load thermal image:\n{thermal_record['path']}\n\n{e}"
            )
            return

        self.display_thermal_image()
        self.update_sync_info_match(
            rgb_record,
            thermal_record,
            item["delta_seconds"],
            item["within_tolerance"]
        )

    # --------------------------------------------------
    # Image loading
    # --------------------------------------------------
    def load_image_file(self, file_path):
        ext = file_path.suffix.lower()

        if ext == ".npy":
            arr = np.load(file_path)
            arr = np.squeeze(arr)

            if arr.ndim != 2:
                raise ValueError(f"Expected 2D .npy thermal image, got shape {arr.shape}")

            arr = arr.astype(np.float32)

            min_val = np.nanmin(arr)
            max_val = np.nanmax(arr)

            if max_val == min_val:
                normalized = np.zeros_like(arr, dtype=np.float32)
            else:
                normalized = (arr - min_val) / (max_val - min_val)

            colored = cm.inferno(normalized)
            rgb = (colored[:, :, :3] * 255).astype(np.uint8)

            return Image.fromarray(rgb)

        return Image.open(file_path).convert("RGB")

    # --------------------------------------------------
    # Display
    # --------------------------------------------------
    def display_rgb_image(self):
        if self.current_rgb_img is None:
            return

        rgb_record = self.rgb_records_filtered[self.rgb_index]
        tag_status = "TAGGED" if self.is_rgb_tagged(rgb_record) else "not tagged"

        img_to_show = self.current_rgb_img.rotate(
            self.rgb_rotation_angle,
            expand=True
        )

        self.rgb_fig.clear()
        self.rgb_ax = self.rgb_fig.add_subplot(111)

        self.rgb_ax.imshow(img_to_show)
        self.rgb_ax.axis("off")
        self.rgb_ax.set_title(
            (
                f"RGB | RFID={rgb_record['rfid']} | Rotation={self.rgb_rotation_angle}° | {tag_status}\n"
                f"{rgb_record['filename']}"
            ),
            fontsize=9
        )

        self.rgb_canvas.draw()

    def display_thermal_image(self):
        if self.current_thermal_img is None or not self.thermal_candidates:
            return

        item = self.thermal_candidates[self.selected_thermal_index]
        thermal_record = item["record"]
        delta = item["delta_seconds"]
        within = item["within_tolerance"]

        status = "WITHIN TOLERANCE" if within else "NEAREST OUTSIDE TOLERANCE"

        img_to_show = self.current_thermal_img.rotate(
            self.thermal_rotation_angle,
            expand=True
        )

        self.thermal_fig.clear()
        self.thermal_ax = self.thermal_fig.add_subplot(111)

        self.thermal_ax.imshow(img_to_show)
        self.thermal_ax.axis("off")
        self.thermal_ax.set_title(
            (
                f"Thermal | {status} | Δt={delta:+.3f} sec | Rotation={self.thermal_rotation_angle}°\n"
                f"{thermal_record['filename']}"
            ),
            fontsize=9
        )

        self.thermal_canvas.draw()

    def display_no_thermal_image(self):
        self.thermal_fig.clear()
        self.thermal_ax = self.thermal_fig.add_subplot(111)

        self.thermal_ax.text(
            0.5,
            0.5,
            "No thermal images loaded\nor no candidates found",
            ha="center",
            va="center",
            fontsize=14
        )

        self.thermal_ax.axis("off")
        self.thermal_canvas.draw()

    def clear_images(self):
        self.current_rgb_img = None
        self.current_thermal_img = None

        self.rgb_fig.clear()
        self.rgb_canvas.draw()

        self.thermal_fig.clear()
        self.thermal_canvas.draw()

        self.thermal_listbox.delete(0, tk.END)

    # --------------------------------------------------
    # Rotation controls
    # --------------------------------------------------
    def rotate_rgb(self, angle):
        if self.current_rgb_img is None:
            return

        self.rgb_rotation_angle = (self.rgb_rotation_angle + angle) % 360
        self.display_rgb_image()
        self.update_tag_info()

    def rotate_thermal(self, angle):
        if self.current_thermal_img is None:
            return

        self.thermal_rotation_angle = (self.thermal_rotation_angle + angle) % 360
        self.display_thermal_image()
        self.update_tag_info()

    def reset_rotation(self):
        self.rgb_rotation_angle = 0
        self.thermal_rotation_angle = 0

        if self.current_rgb_img is not None:
            self.display_rgb_image()

        if self.current_thermal_img is not None:
            self.display_thermal_image()

        self.update_tag_info()

    # --------------------------------------------------
    # Tagging
    # --------------------------------------------------
    def is_rgb_tagged(self, rgb_record):
        return str(rgb_record["path"]) in self.tagged_pairs

    def tag_current_pair(self):
        if not self.rgb_records_filtered:
            messagebox.showerror("No RGB image", "No RGB image is selected.")
            return

        if not self.thermal_candidates:
            messagebox.showerror(
                "No thermal pair",
                "No thermal candidate is selected for this RGB image."
            )
            return

        rgb_record = self.rgb_records_filtered[self.rgb_index]
        item = self.thermal_candidates[self.selected_thermal_index]
        thermal_record = item["record"]

        key = str(rgb_record["path"])

        self.tagged_pairs[key] = {
            "rgb_record": rgb_record,
            "thermal_record": thermal_record,
            "delta_seconds": item["delta_seconds"],
            "within_tolerance": item["within_tolerance"],
            "rgb_rotation_angle": self.rgb_rotation_angle,
            "thermal_rotation_angle": self.thermal_rotation_angle,
            "tagged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.populate_rgb_list()
        self.update_rgb_selection()
        self.display_rgb_image()
        self.update_tag_info()

        messagebox.showinfo(
            "Tagged",
            (
                "Tagged current RGB–thermal pair.\n\n"
                f"RGB:\n{rgb_record['filename']}\n\n"
                f"Thermal:\n{thermal_record['filename']}\n\n"
                f"Δt = {item['delta_seconds']:+.3f} sec"
            )
        )

    def untag_current_rgb(self):
        if not self.rgb_records_filtered:
            return

        rgb_record = self.rgb_records_filtered[self.rgb_index]
        key = str(rgb_record["path"])

        if key not in self.tagged_pairs:
            messagebox.showinfo("Not tagged", "This RGB image is not currently tagged.")
            return

        del self.tagged_pairs[key]

        self.populate_rgb_list()
        self.update_rgb_selection()
        self.display_rgb_image()
        self.update_tag_info()

        messagebox.showinfo("Untagged", "Removed tag from current RGB image.")

    def update_tag_info(self):
        current_status = ""

        if self.rgb_records_filtered:
            rgb_record = self.rgb_records_filtered[self.rgb_index]
            if self.is_rgb_tagged(rgb_record):
                tagged = self.tagged_pairs[str(rgb_record["path"])]
                current_status = (
                    f" | Current RGB is TAGGED with thermal: "
                    f"{tagged['thermal_record']['filename']} | "
                    f"Δt={tagged['delta_seconds']:+.3f} sec"
                )
            else:
                current_status = " | Current RGB is not tagged"

        self.tag_info_label.config(
            text=f"Tagged pairs: {len(self.tagged_pairs)}{current_status}"
        )

    # --------------------------------------------------
    # Export tagged pairs
    # --------------------------------------------------
    def export_tagged_pairs(self):
        if not self.tagged_pairs:
            messagebox.showerror("No tagged pairs", "No RGB–thermal pairs have been tagged yet.")
            return

        output_folder = filedialog.askdirectory(
            title="Select output folder for tagged RGB–thermal dataset"
        )

        if not output_folder:
            return

        output_folder = Path(output_folder)

        rgb_out = output_folder / "RGB"
        thermal_out = output_folder / "Thermal"

        rgb_out.mkdir(parents=True, exist_ok=True)
        thermal_out.mkdir(parents=True, exist_ok=True)

        manifest_path = output_folder / "manifest.csv"

        rows = []

        for idx, item in enumerate(self.tagged_pairs.values(), start=1):
            rgb_record = item["rgb_record"]
            thermal_record = item["thermal_record"]

            pair_id = f"pair_{idx:04d}"

            rgb_suffix = rgb_record["path"].suffix
            thermal_suffix = thermal_record["path"].suffix

            rgb_new_name = f"{pair_id}_RGB_{rgb_record['filename']}"
            thermal_new_name = f"{pair_id}_THERMAL_{thermal_record['filename']}"

            rgb_dest = rgb_out / rgb_new_name
            thermal_dest = thermal_out / thermal_new_name

            try:
                shutil.copy2(rgb_record["path"], rgb_dest)
                shutil.copy2(thermal_record["path"], thermal_dest)
            except Exception as e:
                messagebox.showerror(
                    "Export error",
                    f"Could not copy files for {pair_id}.\n\n{e}"
                )
                return

            rows.append({
                "pair_id": pair_id,
                "rfid_rgb": rgb_record["rfid"],
                "rfid_thermal": thermal_record["rfid"],
                "rgb_original_path": str(rgb_record["path"]),
                "thermal_original_path": str(thermal_record["path"]),
                "rgb_exported_file": str(rgb_dest),
                "thermal_exported_file": str(thermal_dest),
                "rgb_filename": rgb_record["filename"],
                "thermal_filename": thermal_record["filename"],
                "rgb_datetime": rgb_record["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                "thermal_datetime": thermal_record["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                "delta_seconds_thermal_minus_rgb": f"{item['delta_seconds']:.6f}",
                "within_tolerance": item["within_tolerance"],
                "rgb_rotation_angle_for_visual_inspection": item["rgb_rotation_angle"],
                "thermal_rotation_angle_for_visual_inspection": item["thermal_rotation_angle"],
                "rgb_metadata": rgb_record["metadata"],
                "thermal_metadata": thermal_record["metadata"],
                "tagged_at": item["tagged_at"]
            })

        try:
            with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "pair_id",
                    "rfid_rgb",
                    "rfid_thermal",
                    "rgb_original_path",
                    "thermal_original_path",
                    "rgb_exported_file",
                    "thermal_exported_file",
                    "rgb_filename",
                    "thermal_filename",
                    "rgb_datetime",
                    "thermal_datetime",
                    "delta_seconds_thermal_minus_rgb",
                    "within_tolerance",
                    "rgb_rotation_angle_for_visual_inspection",
                    "thermal_rotation_angle_for_visual_inspection",
                    "rgb_metadata",
                    "thermal_metadata",
                    "tagged_at"
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        except Exception as e:
            messagebox.showerror(
                "Manifest error",
                f"Files were copied, but manifest.csv could not be written.\n\n{e}"
            )
            return

        messagebox.showinfo(
            "Export complete",
            (
                f"Exported {len(rows)} tagged RGB–thermal pairs.\n\n"
                f"RGB folder:\n{rgb_out}\n\n"
                f"Thermal folder:\n{thermal_out}\n\n"
                f"Manifest:\n{manifest_path}"
            )
        )

    # --------------------------------------------------
    # Info
    # --------------------------------------------------
    def update_sync_info_match(self, rgb_record, thermal_record, delta_seconds, within):
        rgb_dt = rgb_record["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f")
        thermal_dt = thermal_record["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f")

        tolerance = self.get_tolerance_seconds()
        status = "MATCH within tolerance" if within else "Nearest thermal is OUTSIDE tolerance"

        self.sync_info_label.config(
            text=(
                f"{status} | "
                f"RGB {self.rgb_index + 1}/{len(self.rgb_records_filtered)} | "
                f"RGB RFID: {rgb_record['rfid']} | Thermal RFID: {thermal_record['rfid']} | "
                f"RGB time: {rgb_dt} | Thermal time: {thermal_dt} | "
                f"Δt = Thermal - RGB = {delta_seconds:+.3f} sec | "
                f"Tolerance: ±{tolerance} sec | "
                f"Candidates shown: {len(self.thermal_candidates)}"
            )
        )

    def update_sync_info_no_match(self):
        if not self.rgb_records_filtered:
            return

        rgb_record = self.rgb_records_filtered[self.rgb_index]
        rgb_dt = rgb_record["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f")

        tolerance = self.get_tolerance_seconds()

        if tolerance is None:
            tolerance = 0

        self.sync_info_label.config(
            text=(
                f"RGB {self.rgb_index + 1}/{len(self.rgb_records_filtered)} | "
                f"RFID: {rgb_record['rfid']} | "
                f"RGB time: {rgb_dt} | "
                f"No thermal candidate found. "
                f"Tolerance: ±{tolerance} sec. "
                f"Thermal files loaded: {len(self.thermal_records_all)}"
            )
        )

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------
    def next_rgb(self):
        if not self.rgb_records_filtered:
            return

        self.rgb_index = (self.rgb_index + 1) % len(self.rgb_records_filtered)
        self.load_selected_rgb_and_find_thermal()

    def previous_rgb(self):
        if not self.rgb_records_filtered:
            return

        self.rgb_index = (self.rgb_index - 1) % len(self.rgb_records_filtered)
        self.load_selected_rgb_and_find_thermal()

    # --------------------------------------------------
    # Quit
    # --------------------------------------------------
    def quit_app(self):
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RGBThermalSyncViewer(root)
    root.mainloop()
