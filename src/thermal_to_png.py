import os
from pathlib import Path
from PIL import Image
import matplotlib.cm as cm

import cv2
import numpy as np

"""
This script converts the thermal .npy files into viewable png files that can be passed through
the image processing pipeline. Update the input and output folders below to where the thermal images
are coming from and the output folder you would like them to go to.
"""

# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing the .npy files
INPUT_FOLDER = Path(
    "../Datasets/2026-08-14-all-weights-thermal-npy"
)

# Folder where the converted PNG files will be saved
OUTPUT_FOLDER = Path(
    "../Datasets/GabaGhool2"
)


# ------------------------------------------------------------
# FLIR Ax5 temperature conversion
# ------------------------------------------------------------

# Your camera is a FLIR Ax5 configured with:
#   PixelFormat = Mono14
#   TemperatureLinearMode = On
#   TemperatureLinearResolution = High
#
# FLIR specifies:
#
#     Kelvin = pixel_value * 0.04
#     Celsius = pixel_value * 0.04 - 273.15
#
TEMP_SCALE = 0.04
KELVIN_OFFSET = 273.15


# ------------------------------------------------------------
# Temperature range used to create PNG pixel values
# ------------------------------------------------------------

# Every image will use the SAME temperature range.
#
# This is important for machine-learning applications.
#
# Anything at or below TEMP_MIN_C becomes black.
# Anything at or above TEMP_MAX_C becomes white.
#
# Adjust these values if necessary after inspecting your data.

TEMP_MIN_C = 15.0
TEMP_MAX_C = 40.0


# ------------------------------------------------------------
# Color settings
# ------------------------------------------------------------

# False:
#     Creates a grayscale thermal image.
#
# True:
#     Applies OpenCV's JET thermal-style color map.
#
# I recommend starting with False unless your existing YOLO
# and DeepLabCut training images are colorized thermal images.

USE_COLORMAP = False

# Only used if USE_COLORMAP = True
COLORMAP = cv2.COLORMAP_JET


# ============================================================
# FUNCTIONS
# ============================================================

def npy_to_temperature(data):
    """
    Convert FLIR Ax5 temperature-linear pixel values into
    degrees Celsius.

    FLIR Ax5:
        TemperatureLinearResolution = High
        temperature_K = pixel_value * 0.04

    Therefore:
        temperature_C = pixel_value * 0.04 - 273.15
    """

    temperature_c = (
        data.astype(np.float32) * TEMP_SCALE
        - KELVIN_OFFSET
    )

    return temperature_c


def temperature_to_png(temperature_c):
    """
    Convert a temperature image in degrees Celsius into
    an 8-bit PNG representation using a fixed temperature
    range.

    TEMP_MIN_C -> 0
    TEMP_MAX_C -> 255
    """

    # Clip temperatures outside our display range
    clipped = np.clip(
        temperature_c,
        TEMP_MIN_C,
        TEMP_MAX_C
    )

    # Convert temperature range to 0-255
    normalized = (
        (clipped - TEMP_MIN_C)
        / (TEMP_MAX_C - TEMP_MIN_C)
        * 255.0
    )

    image_8bit = normalized.astype(np.uint8)

    if USE_COLORMAP:
        image_8bit = cv2.applyColorMap(
            image_8bit,
            COLORMAP
        )

    else:
        # OpenCV writes grayscale images correctly as single-channel
        # PNGs. However, making them 3-channel BGR can be useful for
        # compatibility with YOLO and other computer-vision software.
        image_8bit = cv2.cvtColor(
            image_8bit,
            cv2.COLOR_GRAY2BGR
        )

    return image_8bit


def convert_file(input_path, output_path):
    """
    Convert one .npy file to .png.
    """

    print(f"\nProcessing: {input_path.name}")

    # --------------------------------------------------------
    # Load NumPy array
    # --------------------------------------------------------

    data = np.load(input_path)

    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Raw min: {data.min()}")
    print(f"  Raw max: {data.max()}")

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D thermal image, "
            f"but received shape {data.shape}"
        )

    # --------------------------------------------------------
    # Convert FLIR values to Celsius
    # --------------------------------------------------------

    temperature_c = npy_to_temperature(data)

    print(
        f"  Temperature min: "
        f"{temperature_c.min():.2f} °C"
    )

    print(
        f"  Temperature max: "
        f"{temperature_c.max():.2f} °C"
    )

    print(
        f"  Temperature mean: "
        f"{temperature_c.mean():.2f} °C"
    )

    # --------------------------------------------------------
    # Convert temperature to PNG
    # --------------------------------------------------------

    png_image = temperature_to_png(
        temperature_c
    )

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(output_path),
        png_image
    )

    if not success:
        raise RuntimeError(
            f"Failed to save PNG:\n{output_path}"
        )

    print(f"  Saved: {output_path.name}")

def load_image_file(input_path, output_path):

    arr = np.load(input_path)
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

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    Image.fromarray(rgb).save(output_path)

    print(f"  Saved: {output_path.name}")


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check input folder
    # --------------------------------------------------------

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder does not exist:\n{INPUT_FOLDER}"
        )

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find .npy files
    # --------------------------------------------------------

    npy_files = sorted(
        INPUT_FOLDER.glob("*.npy")
    )

    print("=" * 60)
    print("FLIR Ax5 .NPY → .PNG CONVERTER")
    print("=" * 60)

    print(f"Input folder:")
    print(f"  {INPUT_FOLDER}")

    print(f"\nOutput folder:")
    print(f"  {OUTPUT_FOLDER}")

    print(f"\nTemperature conversion:")
    print(f"  Pixel × {TEMP_SCALE} - {KELVIN_OFFSET}")

    print(f"\nPNG temperature range:")
    print(f"  {TEMP_MIN_C} °C → black")
    print(f"  {TEMP_MAX_C} °C → white")

    print(f"\nColor map:")
    print(
        f"  {'JET' if USE_COLORMAP else 'Grayscale'}"
    )

    print(f"\nFound {len(npy_files)} .npy files.")

    if len(npy_files) == 0:
        print("\nNo .npy files found.")
        return

    # --------------------------------------------------------
    # Convert files
    # --------------------------------------------------------

    successful = 0
    failed = 0

    for input_path in npy_files:

        # Keep the original filename, replacing .npy
        # with .png
        output_path = OUTPUT_FOLDER / (
            input_path.stem + ".png"
        )

        try:

            # Keep convert_file to have greyscale thermal images
            # Keep load_image_file to have inferno (normal) thermal images
            # Comment out the version not in use

            load_image_file(
                input_path,
                output_path
            )

            #convert_file(
            #    input_path,
            #    output_path
            #)

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\n  ERROR processing "
                f"{input_path.name}:"
            )

            print(f"  {e}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(f"Successfully converted: {successful}")
    print(f"Failed:                 {failed}")
    print(f"Output folder:          {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()