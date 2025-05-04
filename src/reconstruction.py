# src/reconstruction.py
import numpy as np
import logging
from skimage.transform import iradon, radon, rescale # iradon is for reconstruction
from skimage.data import shepp_logan_phantom # For creating a test phantom
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reconstruct_fbp(
    sinograms: np.ndarray,
    angles_rad: np.ndarray,
    filter_name: str = 'ramp', # Common FBP filter, others: 'shepp-logan', 'cosine', 'hamming', 'hann'
    output_size: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Reconstructs a 3D volume from sinograms using the Filtered Back Projection (FBP) algorithm.

    Reconstructs slice by slice along the height dimension (axis 1).

    Args:
        sinograms (np.ndarray): The preprocessed projection data (attenuation data).
                                Expected shape: (num_projections, height, width).
        angles_rad (np.ndarray): The projection angles in radians. Expected shape: (num_projections,).
        filter_name (str): The filter to use in FBP ('ramp', 'shepp-logan', etc.).
        output_size (Optional[int]): The desired size of the output image slices (width and depth).
                                     If None, it defaults to the width of the sinogram.

    Returns:
        Optional[np.ndarray]: The reconstructed 3D volume with shape (height, output_size, output_size),
                              or None if an error occurs.
    """
    if not isinstance(sinograms, np.ndarray) or sinograms.ndim != 3:
        logging.error("Invalid sinograms input. Expected a 3D NumPy array.")
        return None
    if not isinstance(angles_rad, np.ndarray) or angles_rad.ndim != 1:
        logging.error("Invalid angles input. Expected a 1D NumPy array.")
        return None
    if sinograms.shape[0] != len(angles_rad):
        logging.error(f"Mismatch between number of projections in sinogram ({sinograms.shape[0]}) "
                      f"and number of angles ({len(angles_rad)}).")
        return None

    num_projections, height, width = sinograms.shape
    logging.info(f"Starting FBP reconstruction for volume with dimensions: "
                 f"Projections={num_projections}, Height={height}, Width={width}")

    # Convert angles from radians to degrees for scikit-image's iradon
    angles_deg = np.rad2deg(angles_rad)

    # Determine output size if not specified
    if output_size is None:
        output_size = width
        logging.info(f"Output size not specified, defaulting to sinogram width: {output_size}")

    # Initialize the 3D volume array
    # The reconstructed slices will have shape (output_size, output_size)
    reconstructed_volume = np.zeros((height, output_size, output_size), dtype=np.float32)

    logging.info(f"Reconstructing {height} slices...")

    # Reconstruct slice by slice along the height dimension
    for i in range(height):
        # Extract the 2D sinogram for the current slice
        # Shape needs to be (num_projections, width) for iradon
        sino_slice = sinograms[:, i, :]

        try:
            # Perform Filtered Back Projection on the slice
            # Note: iradon expects theta in degrees
            recon_slice = iradon(
                sino_slice,
                theta=angles_deg,
                output_size=output_size,
                filter_name=filter_name,
                circle=True # Assume the reconstruction region is circular within the square
            )
            reconstructed_volume[i, :, :] = recon_slice.astype(np.float32)

            if i % 50 == 0: # Log progress periodically
                logging.info(f"Reconstructed slice {i+1}/{height}")

        except Exception as e:
            logging.error(f"Error reconstructing slice {i}: {e}")
            # Optionally decide whether to continue or abort
            return None # Abort on error

    logging.info("FBP reconstruction finished.")
    logging.info(f"Reconstructed volume shape: {reconstructed_volume.shape}")
    return reconstructed_volume

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    logging.info("Testing reconstruction module...")

    # 1. Create a phantom
    img_size = 128
    phantom = shepp_logan_phantom()
    phantom = rescale(phantom, scale=img_size / 400.0, mode='reflect', channel_axis=None) # Rescale to desired size

    # 2. Create sinogram (Simulate data acquisition using Radon transform)
    num_projections = 180 # Number of angles/projections
    angles_deg = np.linspace(0., 180., num_projections, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    # Simulate a 3D sinogram by stacking the 2D sinogram multiple times
    # (In reality, each slice's sinogram would be different)
    height = 10 # Simulate a volume height of 10 slices
    dummy_sinogram_2d = radon(phantom, theta=angles_deg, circle=True) # Shape (num_projections, img_size)
    # Stack to create a dummy 3D sinogram (projections, height, width)
    # Note: Need to transpose the 2D sinogram first to match expected input shape
    dummy_sinograms_3d = np.stack([dummy_sinogram_2d.T] * height, axis=1) # Shape (img_size, height, num_projections) ?? No, radon output is (n_angles, n_detectors)
    # radon output is (n_detectors, n_angles) if circle=False, but (n_angles, n_detectors) if circle=True? Check docs.
    # skimage.transform.radon documentation: Output shape is (len(theta), image.shape[1]) -> (num_projections, width)
    # So dummy_sinogram_2d has shape (num_projections, img_size)
    # We need (num_projections, height, width) -> width should be img_size
    dummy_sinograms_3d = np.zeros((num_projections, height, img_size), dtype=np.float32)
    for i in range(height):
        # Add slight variation per slice for realism (optional)
        noise = np.random.normal(0, 0.01 * np.max(dummy_sinogram_2d), dummy_sinogram_2d.shape)
        dummy_sinograms_3d[:, i, :] = (dummy_sinogram_2d + noise).astype(np.float32)

    logging.info(f"Created dummy sinogram data. Shape: {dummy_sinograms_3d.shape}") # (180, 10, 128)
    logging.info(f"Using angles (radians). Shape: {angles_rad.shape}") # (180,)

    # --- Test Reconstruction ---
    logging.info(f"\n--- Testing reconstruct_fbp ---")
    reconstructed_volume = reconstruct_fbp(
        dummy_sinograms_3d,
        angles_rad,
        output_size=img_size # Reconstruct to original phantom size
    )

    if reconstructed_volume is not None:
        logging.info(f"Reconstruction successful. Output volume shape: {reconstructed_volume.shape}")
        # Basic checks
        assert reconstructed_volume.shape == (height, img_size, img_size)
        assert reconstructed_volume.dtype == np.float32

        # Optional: Visualize a slice using matplotlib
        try:
            import matplotlib.pyplot as plt
            middle_slice_index = height // 2
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(phantom, cmap='gray')
            plt.title(f"Original Phantom ({img_size}x{img_size})")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_volume[middle_slice_index, :, :], cmap='gray')
            plt.title(f"Reconstructed Slice {middle_slice_index} (FBP)")
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            # plt.savefig("reconstruction_test.png") # Optionally save the figure
            # logging.info("Saved comparison plot to reconstruction_test.png")
        except ImportError:
            logging.warning("Matplotlib not found. Skipping visualization.")
        except Exception as viz_e:
             logging.error(f"Error during visualization: {viz_e}")

    else:
        logging.error("Reconstruction failed.")

    logging.info("\nReconstruction tests finished.")
