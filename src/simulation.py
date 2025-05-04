# src/simulation.py
import numpy as np
import logging
from skimage.transform import radon, rescale # radon is for forward projection
from skimage.data import shepp_logan_phantom # Standard 2D test phantom
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_3d_phantom(
    phantom_type: str = 'shepp_logan',
    size: int = 128,
    height: int = 10
) -> Optional[np.ndarray]:
    """
    Creates a simple 3D phantom.

    Currently supports stacking a rescaled 2D Shepp-Logan phantom.

    Args:
        phantom_type (str): Type of phantom ('shepp_logan').
        size (int): The width and depth dimension of the phantom slices.
        height (int): The height dimension (number of slices) of the phantom.

    Returns:
        Optional[np.ndarray]: The 3D phantom array with shape (height, size, size),
                              or None if type is unsupported.
    """
    logging.info(f"Creating 3D phantom of type '{phantom_type}' with size={size}, height={height}...")
    if phantom_type.lower() == 'shepp_logan':
        try:
            # Create the standard 2D Shepp-Logan phantom (original size 400x400)
            phantom_2d = shepp_logan_phantom()
            # Rescale it to the desired slice size
            phantom_2d_rescaled = rescale(
                phantom_2d,
                scale=size / 400.0,
                mode='reflect',
                channel_axis=None # For compatibility with newer skimage versions
            )
            # Stack the 2D phantom to create a 3D volume
            # (This creates a uniform phantom along the height axis)
            phantom_3d = np.stack([phantom_2d_rescaled] * height, axis=0)
            logging.info(f"Successfully created 3D Shepp-Logan phantom. Shape: {phantom_3d.shape}")
            return phantom_3d.astype(np.float32)
        except Exception as e:
            logging.error(f"Error creating Shepp-Logan phantom: {e}")
            return None
    else:
        logging.error(f"Unsupported phantom type: {phantom_type}")
        return None

def simulate_projections(
    phantom_3d: np.ndarray,
    angles_rad: np.ndarray
) -> Optional[np.ndarray]:
    """
    Simulates the CT data acquisition process (forward projection) using the Radon transform.

    Projects slice by slice along the height dimension (axis 0).

    Args:
        phantom_3d (np.ndarray): The 3D phantom volume. Expected shape: (height, size, size).
        angles_rad (np.ndarray): The projection angles in radians. Expected shape: (num_projections,).

    Returns:
        Optional[np.ndarray]: The simulated 3D sinogram data with shape (num_projections, height, size),
                              or None if an error occurs.
    """
    if not isinstance(phantom_3d, np.ndarray) or phantom_3d.ndim != 3:
        logging.error("Invalid phantom input. Expected a 3D NumPy array.")
        return None
    if not isinstance(angles_rad, np.ndarray) or angles_rad.ndim != 1:
        logging.error("Invalid angles input. Expected a 1D NumPy array.")
        return None

    height, size, _ = phantom_3d.shape
    num_projections = len(angles_rad)
    logging.info(f"Starting projection simulation for phantom shape {phantom_3d.shape} "
                 f"at {num_projections} angles.")

    # Convert angles to degrees for scikit-image's radon function
    angles_deg = np.rad2deg(angles_rad)

    # Initialize the 3D sinogram array
    # Shape: (num_projections, height, width/size)
    # Note: radon output shape is (num_angles, num_detectors/width)
    simulated_sinograms = np.zeros((num_projections, height, size), dtype=np.float32)

    logging.info(f"Simulating projections for {height} slices...")

    # Simulate slice by slice along the height dimension
    for i in range(height):
        # Extract the 2D slice from the phantom
        phantom_slice = phantom_3d[i, :, :]

        try:
            # Calculate the Radon transform for the current slice
            # Note: radon expects theta in degrees
            sino_slice_2d = radon(
                phantom_slice,
                theta=angles_deg,
                circle=True # Use circle=True if the object is centered and fits within the image circle
            ) # Output shape: (num_projections, size)

            # Store the 2D sinogram in the correct position in the 3D array
            # We need to transpose sino_slice_2d if circle=False, but not if circle=True
            # Based on skimage docs, circle=True output is (len(theta), image.shape[1]) -> (num_projections, size)
            simulated_sinograms[:, i, :] = sino_slice_2d.astype(np.float32)

            if i % 50 == 0: # Log progress periodically
                logging.info(f"Simulated projections for slice {i+1}/{height}")

        except Exception as e:
            logging.error(f"Error simulating projections for slice {i}: {e}")
            return None # Abort on error

    logging.info("Projection simulation finished.")
    logging.info(f"Simulated sinogram shape: {simulated_sinograms.shape}")
    return simulated_sinograms

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    logging.info("Testing simulation module...")

    # --- Parameters ---
    phantom_size = 64 # Smaller size for faster testing
    phantom_height = 5
    num_proj = 90 # Number of projections/angles

    # --- Test Phantom Creation ---
    logging.info(f"\n--- Testing create_3d_phantom ---")
    test_phantom = create_3d_phantom(size=phantom_size, height=phantom_height)

    if test_phantom is not None:
        logging.info(f"Phantom creation successful. Shape: {test_phantom.shape}")
        assert test_phantom.shape == (phantom_height, phantom_size, phantom_size)
        assert test_phantom.dtype == np.float32

        # --- Test Projection Simulation ---
        logging.info(f"\n--- Testing simulate_projections ---")
        test_angles_deg = np.linspace(0., 180., num_proj, endpoint=False)
        test_angles_rad = np.deg2rad(test_angles_deg)

        simulated_sinograms = simulate_projections(test_phantom, test_angles_rad)

        if simulated_sinograms is not None:
            logging.info(f"Simulation successful. Output sinogram shape: {simulated_sinograms.shape}")
            # Basic checks
            assert simulated_sinograms.shape == (num_proj, phantom_height, phantom_size)
            assert simulated_sinograms.dtype == np.float32

            # Optional: Visualize the middle slice of the phantom and its sinogram
            try:
                import matplotlib.pyplot as plt
                middle_slice_index = phantom_height // 2

                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(test_phantom[middle_slice_index, :, :], cmap='gray')
                plt.title(f"Phantom Slice {middle_slice_index}")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.colorbar(label="Attenuation")

                plt.subplot(1, 2, 2)
                # Display the sinogram for the middle slice
                # Sinogram shape is (num_proj, height, size), so we need [:, middle_slice_index, :]
                plt.imshow(simulated_sinograms[:, middle_slice_index, :], cmap='gray', aspect='auto')
                plt.title(f"Simulated Sinogram (Slice {middle_slice_index})")
                plt.xlabel("Detector position (pixels)")
                plt.ylabel("Projection angle (index)")
                plt.colorbar(label="Line Integral")


                plt.tight_layout()
                plt.show()
                # plt.savefig("simulation_test.png")
                # logging.info("Saved comparison plot to simulation_test.png")

            except ImportError:
                logging.warning("Matplotlib not found. Skipping visualization.")
            except Exception as viz_e:
                logging.error(f"Error during visualization: {viz_e}")
        else:
            logging.error("Projection simulation failed.")
    else:
        logging.error("Phantom creation failed.")

    logging.info("\nSimulation tests finished.")
