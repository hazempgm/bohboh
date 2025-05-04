# src/preprocess.py
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def average_images(images: np.ndarray) -> Optional[np.ndarray]:
    """
    Averages a stack of images along the first axis.

    Args:
        images (np.ndarray): A 3D NumPy array (stack_size, height, width).

    Returns:
        Optional[np.ndarray]: A 2D NumPy array (height, width) representing the average,
                              or None if input is not a 3D array.
    """
    if images is None or images.ndim != 3 or images.shape[0] == 0:
        logging.error("Invalid input for averaging. Expected a non-empty 3D array.")
        return None
    try:
        logging.info(f"Averaging {images.shape[0]} images of shape {images.shape[1:]}...")
        # Calculate the mean along the first axis (stack dimension)
        average = np.mean(images, axis=0, dtype=np.float32) # Use float32 for precision
        logging.info("Averaging complete.")
        return average
    except Exception as e:
        logging.error(f"Error during image averaging: {e}")
        return None

def preprocess_data(
    projections: np.ndarray,
    flats: np.ndarray,
    darks: np.ndarray,
    epsilon: float = 1e-6 # Small value to prevent log(0) and division by zero
) -> Optional[np.ndarray]:
    """
    Performs standard preprocessing: dark subtraction, flat-field correction,
    and negative logarithm transformation.

    Args:
        projections (np.ndarray): Raw projection images (num_projections, height, width).
        flats (np.ndarray): Flat-field images (num_flats, height, width).
        darks (np.ndarray): Dark-field images (num_darks, height, width).
        epsilon (float): A small value to avoid division by zero and log(0).

    Returns:
        Optional[np.ndarray]: The preprocessed data (attenuation sinograms) as a 3D NumPy array,
                              or None if an error occurs.
    """
    if not all(isinstance(arr, np.ndarray) for arr in [projections, flats, darks]):
         logging.error("Inputs must be NumPy arrays.")
         return None
    if projections.ndim != 3 or flats.ndim != 3 or darks.ndim != 3:
        logging.error("Projections, flats, and darks must be 3D arrays.")
        return None
    if projections.shape[1:] != flats.shape[1:] or projections.shape[1:] != darks.shape[1:]:
        logging.error("Height and width dimensions of projections, flats, and darks must match.")
        return None

    logging.info("Starting preprocessing...")

    # 1. Average Flats and Darks
    avg_flat = average_images(flats)
    avg_dark = average_images(darks)

    if avg_flat is None or avg_dark is None:
        logging.error("Failed to average flats or darks.")
        return None

    # Ensure averaged images are float32 for calculations
    avg_flat = avg_flat.astype(np.float32)
    avg_dark = avg_dark.astype(np.float32)
    projections = projections.astype(np.float32)

    # 2. Dark Subtraction and Flat-Field Correction
    logging.info("Performing dark subtraction and flat-field correction...")
    # Calculate the denominator for flat-field correction
    denominator = avg_flat - avg_dark

    # Avoid division by zero or very small numbers
    denominator[denominator <= epsilon] = epsilon
    logging.info(f"Denominator min value after epsilon adjustment: {np.min(denominator)}")

    # Apply correction using broadcasting: (proj - dark) / (flat - dark)
    # Ensure projections is also float32 before subtraction
    normalized_projections = (projections - avg_dark) / denominator

    # Clip values to be > 0 before logarithm, handle potential NaNs/Infs from division
    normalized_projections = np.nan_to_num(normalized_projections, nan=epsilon, posinf=epsilon, neginf=epsilon)
    normalized_projections[normalized_projections <= epsilon] = epsilon
    logging.info(f"Normalized projections min/max: {np.min(normalized_projections)} / {np.max(normalized_projections)}")


    # 3. Negative Logarithm Transformation
    logging.info("Applying negative logarithm...")
    # Calculate attenuation: -log(I/I0) = -log(normalized_projection)
    attenuation_data = -np.log(normalized_projections)

    # Handle potential negative infinities if any normalized value was exactly epsilon after clipping
    attenuation_data[np.isneginf(attenuation_data)] = -np.log(epsilon) # Assign a large positive value
    attenuation_data = np.nan_to_num(attenuation_data) # Catch any remaining NaNs

    logging.info("Preprocessing finished.")
    logging.info(f"Output attenuation data shape: {attenuation_data.shape}, dtype: {attenuation_data.dtype}")
    logging.info(f"Output attenuation data min/max: {np.min(attenuation_data)} / {np.max(attenuation_data)}")


    return attenuation_data.astype(np.float32) # Ensure final output is float32

# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    logging.info("Testing preprocess module...")

    # Create plausible dummy data
    num_proj, h, w = 10, 50, 60
    num_flats, num_darks = 5, 5

    # Darks: low values
    dummy_darks = np.random.rand(num_darks, h, w).astype(np.float32) * 100 + 50
    # Flats: high values, higher than darks
    dummy_flats = np.random.rand(num_flats, h, w).astype(np.float32) * 1000 + np.mean(dummy_darks) + 500
    # Projections: values between darks and flats, simulating attenuation
    avg_dark_val = np.mean(dummy_darks)
    avg_flat_val = np.mean(dummy_flats)
    dummy_projections = np.random.rand(num_proj, h, w).astype(np.float32) * (avg_flat_val - avg_dark_val) + avg_dark_val
    # Simulate some strong attenuation (lower transmission values) in the center
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask = (y - center_h)**2 + (x - center_w)**2 < radius**2
    dummy_projections[:, mask] *= 0.3 # Reduce transmission significantly in the center


    logging.info(f"Dummy Projections shape: {dummy_projections.shape}, min/max: {np.min(dummy_projections):.2f}/{np.max(dummy_projections):.2f}")
    logging.info(f"Dummy Flats shape: {dummy_flats.shape}, min/max: {np.min(dummy_flats):.2f}/{np.max(dummy_flats):.2f}")
    logging.info(f"Dummy Darks shape: {dummy_darks.shape}, min/max: {np.min(dummy_darks):.2f}/{np.max(dummy_darks):.2f}")


    # --- Test Preprocessing ---
    logging.info(f"\n--- Testing preprocess_data ---")
    preprocessed_attenuation = preprocess_data(dummy_projections, dummy_flats, dummy_darks)

    if preprocessed_attenuation is not None:
        logging.info(f"Preprocessing successful. Output shape: {preprocessed_attenuation.shape}")
        logging.info(f"Output min/max attenuation: {np.min(preprocessed_attenuation):.4f} / {np.max(preprocessed_attenuation):.4f}")
        # Basic checks
        assert preprocessed_attenuation.shape == (num_proj, h, w)
        assert preprocessed_attenuation.dtype == np.float32
        assert np.all(np.isfinite(preprocessed_attenuation)) # Check for NaNs or Infs
        # Check that attenuated area has higher values
        avg_attenuation_center = np.mean(preprocessed_attenuation[:, mask])
        avg_attenuation_outside = np.mean(preprocessed_attenuation[:, ~mask])
        logging.info(f"Avg attenuation center: {avg_attenuation_center:.4f}, outside: {avg_attenuation_outside:.4f}")
        assert avg_attenuation_center > avg_attenuation_outside # Center should have higher attenuation values
    else:
        logging.error("Preprocessing failed.")

    logging.info("\nPreprocessing tests finished.")
