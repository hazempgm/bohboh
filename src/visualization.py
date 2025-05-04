# src/visualization.py
import numpy as np
import logging
from typing import Optional, Tuple

# Attempt to import matplotlib, but don't fail if it's not installed
# This allows other modules to import this one even without matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not found. Visualization functions will be unavailable.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_matplotlib():
    """Checks if matplotlib is available and logs a warning if not."""
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib is required for visualization but not installed.")
        return False
    return True

def plot_slice(
    slice_data: np.ndarray,
    title: str = "Slice",
    cmap: str = 'gray',
    colorbar_label: str = "Value"
):
    """
    Displays a single 2D NumPy array as an image.

    Args:
        slice_data (np.ndarray): The 2D array to display.
        title (str): The title for the plot.
        cmap (str): The colormap to use (e.g., 'gray', 'viridis').
        colorbar_label (str): Label for the colorbar.
    """
    if not check_matplotlib(): return
    if not isinstance(slice_data, np.ndarray) or slice_data.ndim != 2:
        logging.error(f"Invalid input for plot_slice. Expected a 2D NumPy array, got shape {slice_data.shape if isinstance(slice_data, np.ndarray) else type(slice_data)}.")
        return

    plt.figure(figsize=(6, 6))
    im = plt.imshow(slice_data, cmap=cmap)
    plt.title(title)
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.colorbar(im, label=colorbar_label)
    plt.tight_layout()
    # Let the caller decide whether to show or save
    # plt.show()

def plot_sinogram(
    sinogram_slice: np.ndarray,
    title: str = "Sinogram Slice",
    cmap: str = 'gray',
    colorbar_label: str = "Line Integral / Attenuation"
):
    """
    Displays a 2D sinogram (projections vs. angles).

    Args:
        sinogram_slice (np.ndarray): The 2D sinogram data.
                                     Expected shape: (num_projections, width).
        title (str): The title for the plot.
        cmap (str): The colormap to use.
        colorbar_label (str): Label for the colorbar.
    """
    if not check_matplotlib(): return
    if not isinstance(sinogram_slice, np.ndarray) or sinogram_slice.ndim != 2:
        logging.error(f"Invalid input for plot_sinogram. Expected a 2D NumPy array, got shape {sinogram_slice.shape if isinstance(sinogram_slice, np.ndarray) else type(sinogram_slice)}.")
        return

    plt.figure(figsize=(8, 5))
    # Use aspect='auto' for typical sinogram appearance
    im = plt.imshow(sinogram_slice, cmap=cmap, aspect='auto')
    plt.title(title)
    plt.xlabel("Detector position (pixel)")
    plt.ylabel("Projection angle (index)")
    plt.colorbar(im, label=colorbar_label)
    plt.tight_layout()
    # plt.show()

def plot_comparison(
    slice1: np.ndarray,
    slice2: np.ndarray,
    title1: str = "Slice 1",
    title2: str = "Slice 2",
    main_title: str = "Comparison",
    cmap: str = 'gray',
    colorbar_label: str = "Value"
):
    """
    Displays two 2D slices side-by-side for comparison.

    Args:
        slice1 (np.ndarray): The first 2D array.
        slice2 (np.ndarray): The second 2D array.
        title1 (str): Title for the first subplot.
        title2 (str): Title for the second subplot.
        main_title (str): Overall title for the figure.
        cmap (str): The colormap to use.
        colorbar_label (str): Label for the colorbars.
    """
    if not check_matplotlib(): return
    if not isinstance(slice1, np.ndarray) or slice1.ndim != 2:
        logging.error(f"Invalid input for slice1. Expected a 2D NumPy array.")
        return
    if not isinstance(slice2, np.ndarray) or slice2.ndim != 2:
        logging.error(f"Invalid input for slice2. Expected a 2D NumPy array.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(main_title)

    # Plot Slice 1
    im1 = axes[0].imshow(slice1, cmap=cmap)
    axes[0].set_title(title1)
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Y pixel")
    fig.colorbar(im1, ax=axes[0], label=colorbar_label, shrink=0.8) # Adjust shrink as needed

    # Plot Slice 2
    im2 = axes[1].imshow(slice2, cmap=cmap)
    axes[1].set_title(title2)
    axes[1].set_xlabel("X pixel")
    # axes[1].set_ylabel("Y pixel") # Optional: remove y-label for second plot
    fig.colorbar(im2, ax=axes[1], label=colorbar_label, shrink=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    # plt.show()


# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    if not check_matplotlib():
        logging.error("Cannot run visualization tests without matplotlib.")
    else:
        logging.info("Testing visualization module...")

        # Create dummy data
        dummy_slice = np.random.rand(64, 64)
        dummy_sinogram = np.random.rand(90, 64) # 90 projections, 64 detector pixels
        dummy_slice_other = dummy_slice * 0.8 + np.random.rand(64, 64) * 0.1

        # --- Test plot_slice ---
        logging.info("\n--- Testing plot_slice ---")
        plot_slice(dummy_slice, title="Test Slice Display")
        plt.show() # Show the plot generated by the function

        # --- Test plot_sinogram ---
        logging.info("\n--- Testing plot_sinogram ---")
        plot_sinogram(dummy_sinogram, title="Test Sinogram Display")
        plt.show()

        # --- Test plot_comparison ---
        logging.info("\n--- Testing plot_comparison ---")
        plot_comparison(dummy_slice, dummy_slice_other,
                        title1="Original Slice", title2="Modified Slice",
                        main_title="Side-by-Side Comparison")
        plt.show()

        logging.info("\nVisualization tests finished. Close plot windows to exit.")
