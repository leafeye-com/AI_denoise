#! /usr/bin/env python3

import cv2
import numpy as np
from network import Network
import os
import argparse
from pathlib import Path

def load(filename: str) -> np.ndarray:
    """
    Load an RGB image from a file.

    Args:
        filename (str): Path to the image file

    Returns:
        np.ndarray: RGB image array
    """
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save(filename: str, image: np.ndarray, overwrite: bool = False) -> None:
    """
    Save an RGB image to a file. You can save a 16-bit PNG, but for JPEG it will be saved as 8-bit.

    Args:
        filename (str): Path to save the image
        image (np.ndarray): RGB image array to save (8 or 16-bit)
        overwrite (bool): Whether to overwrite the output file if it already exists
    """
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists")

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR, not RGB
    if filename.endswith(".jpg") and image.dtype == np.uint16:
        bgr_image = (bgr_image >> 8).astype(np.uint8)
    cv2.imwrite(filename, bgr_image)

def do_denoise(image: np.ndarray, network: Network, overlap_pixels: int = 16, show_progress: bool = True) -> np.ndarray:
    """
    Return the denoised image as a new array.

    Args:
        image (np.ndarray): RGB image array to denoise
        network (Network): Network model to use for denoising
        overlap_pixels (int): Number of overlap pixels between image patches
        show_progress (bool): Whether to show progress bar

    Returns:
        np.ndarray: A copy of the image which has been denoised.
    """
    image = image.astype(np.float32) / 255.0
    denoised = network.run_inference(image, overlap_pixels, show_progress)
    denoised = (denoised * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    return denoised

if __name__ == "__main__":
    """
    Command line interface to denoise an RGB image. Usage:

    python rgb.py --input input.png --output output.jpg
      Simple example where input.png is denoised to make output.jpg with the default network. PNG is
      recommended for the input files.

    python rgb.py --input input.png --output output.jpg --network networks/nafnet_rgb.tflite -y
      As above, but using a named network model. Overwrite the output file if it already exists.

    Type "python rgb.py --help" for more options.
    """
    default_network_path = str(Path(__file__).resolve().parent / "networks" / "nafnet_rgb_small.tflite")

    parser = argparse.ArgumentParser(description="Denoise an RGB image.")
    parser.add_argument("--input", required=True, help="Input RGB image filename.")
    parser.add_argument("--network", default=default_network_path, help=f"Network model filename (default: {default_network_path})")
    parser.add_argument("--overlap", type=int, default=16, help="Number of overlap pixels between image patches")
    parser.add_argument("--output", required=True, help="Output RGB image filename.")
    parser.add_argument("-y", "--yes", action="store_true", help="Overwrite output file if it exists.")
    args = parser.parse_args()

    network = Network(args.network)
    image = load(args.input)
    denoised = do_denoise(image, network, args.overlap)
    save(args.output, denoised, overwrite=args.yes)