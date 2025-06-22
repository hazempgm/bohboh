import numpy as np
from skimage.transform import iradon
from tqdm import tqdm # A library to show progress bars

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
    # The scikit-image iradon function expects the sinogram to be (num_detectors, num_angles).
    # Our data is (num_angles, num_detectors), so we need to transpose it.
    reconstructed = iradon(sinogram.T, theta=angles, circle=True)
    return reconstructed


def reconstruct_full_volume(projection_stack: np.ndarray) -> np.ndarray:
    """
    Reconstructs the full 3D volume from a stack of 2D projection images.

    Args:
        projection_stack (np.ndarray): The stack of 2D projections. 
                                     Shape (num_angles, height, width).

    Returns:
        np.ndarray: The reconstructed 3D volume. Shape (height, recon_size, recon_size).
    """
    print("Starting full 3D reconstruction...")
    
    num_angles, height, width = projection_stack.shape
    
    # Define the angles at which the projections were taken.
    theta = np.linspace(0., 180., num_angles, endpoint=False)
    
    reconstructed_slices = []

    # Use tqdm to show a progress bar during the loop
    for i in tqdm(range(height), desc="Reconstructing slices"):
        # Extract the sinogram for the current slice
        sinogram = projection_stack[:, i, :]
        
        # Reconstruct the slice
        rec_slice = reconstruct_slice(sinogram, theta)
        reconstructed_slices.append(rec_slice)

    print("\nStacking reconstructed slices into a 3D volume...")
    # Stack the list of 2D reconstructed slices into a single 3D numpy array
    reconstructed_volume = np.stack(reconstructed_slices, axis=0)
    
    print("Full 3D reconstruction complete.")
    return reconstructed_volume