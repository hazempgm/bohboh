import os
import json
import tifffile as tiff
import numpy as np
from typing import List, Tuple, Optional
import glob

def _parse_metadata(json_path: str) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    (Internal) Parses the JSON metadata file to extract and SORT the list of
    projection filenames and their corresponding angles.
    """
    if not os.path.exists(json_path):
        print(f"Error: Metadata file not found at '{json_path}'")
        return None
        
    print(f"Parsing metadata from: {json_path}")
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    try:
        filenames = metadata['projections']['images']['files']
        angle_data = metadata['geometry']['projectionAngles']
        # Create a list of (angle, index) tuples for sorting
        angles_with_indices = [(item['angle'], item['index']) for item in angle_data]
    except KeyError as e:
        print(f"Error: Could not find required key in JSON file: {e}")
        return None

    # --- CRITICAL CORRECTION: Sort the data by angle ---
    # The iradon function requires angles to be in increasing order.
    # We must sort both the angles and the filenames together to maintain the correct pairs.
    
    # 1. Pair each original filename with its angle data.
    # We assume the initial lists are correctly aligned by their original order.
    if len(filenames) != len(angles_with_indices):
        print("Warning: Initial mismatch between number of files and angles. This may cause issues.")
        # We proceed cautiously, but this indicates a potential data integrity problem.
        
    # We use the index from the angle_data to align with the filename list, which is 0-indexed.
    # The 'index' in the JSON is 1-based, so we subtract 1.
    try:
        paired_data = [(angles_with_indices[i][0], filenames[angles_with_indices[i][1] - 1]) for i in range(len(angles_with_indices))]
    except IndexError:
        print("Error: Index from metadata is out of bounds for the filenames list. Cannot safely pair data.")
        return None

    # 2. Sort the pairs based on the angle (the first element of the tuple).
    sorted_paired_data = sorted(paired_data, key=lambda x: x[0])
    
    # 3. Unzip the sorted pairs back into separate lists.
    sorted_angles, sorted_filenames = zip(*sorted_paired_data)
    
    print(f"Data sorted by angle. Found {len(sorted_filenames)} filenames and {len(sorted_angles)} angles.")
    
    return list(sorted_filenames), np.array(sorted_angles)

def _load_images_from_list(directory_path: str, file_list: List[str]) -> np.ndarray:
    """(Internal) Loads a specific list of TIFF images from a directory."""
    images: List[np.ndarray] = []
    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        try:
            image = tiff.imread(file_path)
            images.append(image)
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")
    
    return np.stack(images, axis=0) if images else np.array([])


def load_projection_data(directory_path: str, use_metadata: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Loads projection data (images and angles) from a directory.

    If use_metadata is True, it tries to find and parse a .json file to get the
    exact file order and projection angles, then sorts them by angle.

    If use_metadata is False or fails, it loads all tiffs sorted alphabetically
    and generates evenly spaced angles.

    Args:
        directory_path (str): Path to the directory with image data.
        use_metadata (bool): Flag to control whether to use a JSON metadata file.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: A tuple of (image_stack, angles), or None on failure.
    """
    if use_metadata:
        print("Attempting to load data using JSON metadata...")
        json_files = glob.glob(os.path.join(directory_path, '*.json'))
        if not json_files:
            print("Error: 'use_metadata' is True, but no .json file was found.")
            return None
        
        metadata_result = _parse_metadata(json_files[0])
        
        if metadata_result:
            filenames, angles = metadata_result
            image_stack = _load_images_from_list(directory_path, filenames)
            if image_stack.size > 0:
                return image_stack, angles
    
    # Fallback if use_metadata is False or if it failed
    if use_metadata:
        print("Metadata loading failed. Falling back to default method.")
    
    print("Loading data without metadata (sorting files alphabetically)...")
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return None

    all_files = sorted(glob.glob(os.path.join(directory_path, '*.tif*')))
    
    if not all_files:
        print(f"Error: No .tiff files found in '{directory_path}'")
        return None
        
    image_stack = _load_images_from_list(directory_path, [os.path.basename(f) for f in all_files])
    
    if image_stack.size == 0:
        return None
        
    num_projections = image_stack.shape[0]
    print(f"Generating {num_projections} evenly spaced angles from 0 to 180 degrees.")
    angles = np.linspace(0., 180., num_projections, endpoint=False)
    
    return image_stack, angles
