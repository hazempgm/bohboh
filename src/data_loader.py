import os
import tifffile as tiff
import numpy as np
from typing import List

def load_tiff_images(directory_path: str) -> np.ndarray:
    """
    Loads all TIFF images from a specified directory, sorts them by name,
    and stacks them into a single 3D NumPy array.

    Args:
        directory_path (str): The path to the directory containing the .tiff files.

    Returns:
        np.ndarray: A 3D NumPy array containing the stacked images.
                    Returns an empty array if the directory is not found or contains no .tiff files.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return np.array([])

    image_files: List[str] = sorted([
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(('.tif', '.tiff'))
    ])

    if not image_files:
        print(f"Error: No .tiff files found in '{directory_path}'")
        return np.array([])

    images: List[np.ndarray] = []
    for file_path in image_files:
        try:
            image = tiff.imread(file_path)
            images.append(image)
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")
            continue
    
    if not images:
        print("Error: Could not read any of the tiff files successfully.")
        return np.array([])

    print(f"Successfully loaded {len(images)} images.")
    
    # Stack images into a single 3D numpy array
    # The result is a (num_images, height, width) array
    return np.stack(images, axis=0)