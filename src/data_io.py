# src/data_io.py
import tifffile
import numpy as np
from pathlib import Path
import logging
import configparser # Import configparser to read INI-style files
from typing import List, Tuple, Optional, Union

# Configure logging
# Note: This basicConfig might be overridden by utils.setup_logging later
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tiff_sequence(directory_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Loads a sequence of TIFF files from a directory and stacks them into a 3D NumPy array.

    Args:
        directory_path (Union[str, Path]): The path to the directory containing the TIFF files.

    Returns:
        Optional[np.ndarray]: A 3D NumPy array (stack_size, height, width) if successful,
                              None otherwise. Returns None if directory is empty or files
                              cannot be read.
    """
    # Ensure the input is a Path object
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        logging.error(f"Error: Provided path '{directory_path}' is not a valid directory.")
        return None

    # Find all TIFF files, sort them numerically/alphabetically
    # Assumes filenames allow correct sorting (e.g., proj_0000.tiff, proj_0001.tiff)
    tiff_files = sorted(list(directory_path.glob('*.tif*'))) # Handles .tif and .tiff

    if not tiff_files:
        logging.warning(f"No TIFF files found in directory: {directory_path}")
        return None

    logging.info(f"Found {len(tiff_files)} TIFF files in {directory_path}. Loading sequence...")

    try:
        # Read all images using tifffile.imread with pattern matching
        image_stack = tifffile.imread(f"{directory_path}/*.tif*") # Use glob pattern directly
        logging.info(f"Successfully loaded image stack with shape: {image_stack.shape}")
        return image_stack
    except Exception as e:
        logging.error(f"Error loading TIFF sequence from {directory_path}: {e}")
        return None


def load_metadata(metadata_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Loads metadata from an INI-style text file and calculates the projection angles.

    Reads the [ACQUISITION] section to find NumberImages, AngleFirst, AngleInterval.

    Args:
        metadata_path (Union[str, Path]): Path to the metadata INI file.

    Returns:
        Optional[np.ndarray]: A 1D NumPy array of angles in radians if successful, None otherwise.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        logging.error(f"Error: Metadata file not found at '{metadata_path}'")
        return None

    config = configparser.ConfigParser()
    try:
        config.read(metadata_path)

        # Check if the required section exists
        if 'ACQUISITION' not in config:
            logging.error(f"Metadata file '{metadata_path}' is missing the [ACQUISITION] section.")
            return None

        acq_section = config['ACQUISITION']

        # Read necessary values, converting to appropriate types
        num_images = acq_section.getint('NumberImages')
        angle_first_deg = acq_section.getfloat('AngleFirst')
        angle_interval_deg = acq_section.getfloat('AngleInterval')

        logging.info(f"Read from metadata: NumberImages={num_images}, AngleFirst={angle_first_deg}, AngleInterval={angle_interval_deg}")

        # Calculate the sequence of angles in degrees
        # Example: num=4, first=0, interval=1 -> angles are 0, 1, 2, 3
        # endpoint = angle_first_deg + angle_interval_deg * (num_images - 1)
        # angles_deg = np.linspace(angle_first_deg, endpoint, num_images, endpoint=True)

        # Alternative using arange: start, stop (exclusive), step
        angles_deg = np.arange(
            angle_first_deg,
            angle_first_deg + angle_interval_deg * num_images,
            angle_interval_deg,
            dtype=np.float32
        )

        # Verify the number of angles generated matches num_images
        if len(angles_deg) != num_images:
             logging.warning(f"Calculated number of angles ({len(angles_deg)}) does not match NumberImages ({num_images}) specified in metadata. Using calculated angles.")
             # Decide if this should be an error or just a warning. Here, we proceed with calculated angles.

        # Convert angles from degrees to radians, as often required by reconstruction libraries
        angles_rad = np.deg2rad(angles_deg)
        logging.info(f"Successfully calculated {len(angles_rad)} angles (radians) from metadata.")
        return angles_rad

    except configparser.Error as e:
        logging.error(f"Error parsing metadata file {metadata_path}: {e}")
        return None
    except KeyError as e:
         logging.error(f"Missing expected key in [ACQUISITION] section of {metadata_path}: {e}")
         return None
    except ValueError as e:
        logging.error(f"Error converting value in [ACQUISITION] section of {metadata_path} (expecting number): {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred reading metadata file {metadata_path}: {e}")
        return None

def save_tiff_stack(volume: np.ndarray, output_directory: Union[str, Path], file_prefix: str = "recon_slice"):
    """
    Saves a 3D NumPy array as a sequence of 2D TIFF slices in a specified directory.

    Args:
        volume (np.ndarray): The 3D volume data (depth, height, width).
        output_directory (Union[str, Path]): The directory where the TIFF slices will be saved.
        file_prefix (str): The prefix for the output filenames (e.g., "recon_slice").
                           Files will be named like "recon_slice_0000.tiff".
    """
    output_directory = Path(output_directory)
    # Create the output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)

    if volume.ndim != 3:
        logging.error(f"Error: Input volume must be 3-dimensional, but got shape {volume.shape}")
        return

    num_slices = volume.shape[0]
    # Determine padding width based on the number of slices (e.g., 100 slices -> 3 digits, 1000 -> 4)
    zfill_width = len(str(num_slices - 1))

    logging.info(f"Saving {num_slices} slices to {output_directory} with prefix '{file_prefix}'...")

    try:
        for i in range(num_slices):
            slice_data = volume[i, :, :]
            # Format filename with zero-padding (e.g., recon_slice_0001.tiff)
            filename = f"{file_prefix}_{str(i).zfill(zfill_width)}.tiff"
            filepath = output_directory / filename

            # Save the 2D slice as a TIFF file
            # Use appropriate dtype, potentially converting if needed
            tifffile.imwrite(filepath, slice_data.astype(np.float32), imagej=True) # imagej=True helps with Fiji/ImageJ compatibility

            if i % 100 == 0: # Log progress
                 logging.info(f"Saved {i+1}/{num_slices} slices...")

        logging.info(f"Successfully saved TIFF stack to {output_directory}.")
    except Exception as e:
        logging.error(f"Error saving TIFF stack: {e}")