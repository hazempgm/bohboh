import numpy as np
from skimage.transform import iradon

def reconstruct_slice(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Reconstructs a 2D slice from a sinogram using the inverse Radon transform
    (Filtered Back-Projection).

    Args:
        sinogram (np.ndarray): The sinogram to reconstruct. Shape (num_angles, num_detectors).
        angles (np.ndarray): The angles (in degrees) at which the projections were taken.

    Returns:
        np.ndarray: The reconstructed 2D slice.
    """
    print("Reconstructing slice...")
    
    # The scikit-image iradon function expects the sinogram to be (num_detectors, num_angles).
    # Our data is (num_angles, num_detectors), so we need to transpose it.
    reconstructed = iradon(sinogram.T, theta=angles, circle=True)
    
    print("Reconstruction complete.")
    return reconstructed