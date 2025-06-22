import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

def segment_volume_by_threshold(volume: np.ndarray) -> np.ndarray:
    """
    Segments the object from the background in a 3D volume using an
    automatically determined threshold (Otsu's method).

    Args:
        volume (np.ndarray): The reconstructed 3D volume.

    Returns:
        np.ndarray: A binary mask of the same shape as the volume, where
                    True (1) indicates the object and False (0) indicates the background.
    """
    print("Segmenting volume to isolate the object...")
    
    # Otsu's method automatically finds the best threshold to separate
    # the two main classes of voxels (object and background).
    threshold_value = threshold_otsu(volume)
    
    print(f"Automatically determined Otsu threshold: {threshold_value:.4f}")
    
    # Create a binary mask based on the threshold
    mask = volume > threshold_value
    
    return mask

def plot_volume_histogram(volume: np.ndarray):
    """
    Plots a histogram of the voxel intensity values in the 3D volume.

    Args:
        volume (np.ndarray): The 3D volume to analyze.
    """
    print("Plotting histogram of voxel values...")
    plt.figure(figsize=(10, 5))
    plt.hist(volume.flatten(), bins=100, log=True)
    plt.title('Histogram of Reconstructed Voxel Intensities')
    plt.xlabel('Voxel Intensity')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

