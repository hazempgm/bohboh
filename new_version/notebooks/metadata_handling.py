"""
Metadata handling code for tomographic reconstruction.
This file provides utilities to load and use metadata from JSON files
that accompany TIFF images in modern CT systems.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
from pathlib import Path

# Add the src directory to the path
sys.path.append('../src')

# Import our custom modules
from utils import create_projection_geometry

def extract_angles_from_json(metadata_json):
    """
    Extract angles from JSON metadata structure.
    
    Args:
        metadata_json (dict): Loaded JSON metadata
        
    Returns:
        np.ndarray: Array of angles in degrees
    """
    if 'geometry' in metadata_json and 'projectionAngles' in metadata_json['geometry']:
        # Extract angles from the projectionAngles array
        angle_data = metadata_json['geometry']['projectionAngles']
        angles = np.array([item['angle'] for item in angle_data])
        # Convert to degrees if needed
        if np.max(angles) < 7:  # Assume radians if max angle is small
            print("Converting angles from radians to degrees")
            angles = np.rad2deg(angles)
        return angles
    else:
        # Fallback to totalAngle if available
        if 'geometry' in metadata_json and 'totalAngle' in metadata_json['geometry']:
            total_angle = metadata_json['geometry']['totalAngle']
            num_projections = metadata_json['projections']['numProjections']
            print(f"Creating {num_projections} evenly spaced angles over {total_angle} degrees")
            return np.linspace(0, total_angle, num_projections, endpoint=False)
    
    # If no angle information found
    raise ValueError("No angle information found in metadata")

def load_json_metadata(metadata_path):
    """
    Load and parse JSON metadata.
    
    Args:
        metadata_path (str): Path to JSON metadata file
        
    Returns:
        dict: Parsed metadata
    """
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON metadata: {e}")
        return {}

def load_projections_from_files(data_path, file_list, dtype=np.float32):
    """
    Load projection images from a list of files.
    
    Args:
        data_path (str): Base directory for image files
        file_list (list): List of file names to load
        dtype: Data type for the loaded projections
        
    Returns:
        np.ndarray: Loaded projections array
    """
    import tifffile
    
    if not file_list:
        raise ValueError("No projection files found")
    
    # Load first image to get dimensions
    first_img_path = os.path.join(data_path, file_list[0])
    first_img = tifffile.imread(first_img_path)
    
    # Initialize projections array
    projections = np.zeros((len(file_list), first_img.shape[0], first_img.shape[1]), dtype=dtype)
    
    # Load all projections
    print(f"Loading {len(file_list)} projection images...")
    for i, filename in enumerate(file_list):
        img_path = os.path.join(data_path, filename)
        projections[i] = tifffile.imread(img_path)
    
    return projections

def load_data_with_json_metadata(data_path, metadata_filename='metadata.json', pattern="*.tif*"):
    """
    Load projection data with metadata from JSON file.
    
    Args:
        data_path (str): Path to directory with TIFF files and metadata
        metadata_filename (str): Filename of the JSON metadata
        pattern (str): Pattern for TIFF files if not specified in metadata
        
    Returns:
        tuple: (projections, angles, metadata, geometry)
    """
    print(f"Loading projections and metadata from {data_path}...")
    
    # Load JSON metadata
    metadata_path = os.path.join(data_path, metadata_filename)
    if os.path.exists(metadata_path):
        metadata = load_json_metadata(metadata_path)
        print("JSON metadata loaded successfully")
    else:
        print(f"Warning: No metadata file found at {metadata_path}")
        metadata = {}
    
    # Extract projection filenames
    if metadata and 'projections' in metadata and 'images' in metadata['projections'] and 'files' in metadata['projections']['images']:
        # Use filenames from metadata
        file_list = metadata['projections']['images']['files']
        print(f"Found {len(file_list)} projection files in metadata")
    else:
        # Fallback to pattern search
        file_list = sorted([os.path.basename(f) for f in glob.glob(os.path.join(data_path, pattern))])
        print(f"Found {len(file_list)} projection files matching pattern {pattern}")
    
    # Load projection images
    projections = load_projections_from_files(data_path, file_list)
    
    # Extract angles from metadata
    if metadata:
        try:
            angles = extract_angles_from_json(metadata)
            print(f"Extracted {len(angles)} angles from metadata")
        except ValueError as e:
            print(f"Error extracting angles: {e}")
            # Create default angles
            angles = np.linspace(0, 360, len(file_list), endpoint=False)
            print(f"Created {len(angles)} default evenly spaced angles")
    else:
        # Create default angles
        angles = np.linspace(0, 360, len(file_list), endpoint=False)
        print(f"No metadata found. Created {len(angles)} default evenly spaced angles")
    
    # Create geometry dictionary
    geometry = extract_geometry_from_json(metadata, projections.shape)
    
    return projections, angles, metadata, geometry

def extract_geometry_from_json(metadata, proj_shape):
    """
    Extract geometry information from JSON metadata.
    
    Args:
        metadata (dict): Loaded JSON metadata
        proj_shape (tuple): Shape of projections array
        
    Returns:
        dict: Geometry dictionary for reconstruction
    """
    # Default values
    source_origin_dist = 500.0  # mm
    origin_detector_dist = 500.0  # mm
    detector_shape = (proj_shape[1], proj_shape[2])
    
    if metadata and 'geometry' in metadata:
        geom = metadata['geometry']
        
        # Extract distances
        if 'distanceSourceObject' in geom:
            source_origin_dist = float(geom['distanceSourceObject'])
        
        if 'distanceObjectDetector' in geom:
            origin_detector_dist = float(geom['distanceObjectDetector'])
            
        # Extract detector shape if available
        if 'detectorPixel' in geom:
            detector_shape = tuple(geom['detectorPixel'])
        
        # Extract angles
        try:
            angles = extract_angles_from_json(metadata)
        except ValueError:
            angles = np.linspace(0, 360, proj_shape[0], endpoint=False)
        
        print("\nGeometry information from metadata:")
        print(f"  Source-to-Origin distance: {source_origin_dist} mm")
        print(f"  Origin-to-Detector distance: {origin_detector_dist} mm")
        print(f"  Detector shape: {detector_shape}")
        print(f"  Number of projection angles: {len(angles)}")
    else:
        print("\nNo geometry information found in metadata. Using default values.")
        # Create default angles
        angles = np.linspace(0, 360, proj_shape[0], endpoint=False)
    
    # Create geometry dictionary
    geometry = create_projection_geometry(
        angles,
        detector_shape,
        source_origin_dist,
        origin_detector_dist
    )
    
    return geometry

# Example of how to use this in the notebook:
"""
# Use the metadata handling code
from metadata_handling import load_data_with_json_metadata

# Load data with JSON metadata
projections, angles, metadata, geometry = load_data_with_json_metadata(data_path, 'metadata.json')

# Use the data and geometry information for reconstruction
# ...

# You can also access specific parameters from the metadata
if 'geometry' in metadata and 'distanceSourceObject' in metadata['geometry']:
    print(f"Source-object distance: {metadata['geometry']['distanceSourceObject']} mm")
"""
