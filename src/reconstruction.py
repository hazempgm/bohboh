# -*- coding: utf-8 -*-
"""
Module for performing CT reconstruction.

Supports different algorithms based on configuration:
- FBP (Filtered Back Projection) via scikit-image (primarily for 2D parallel beam)
- FDK (Feldkamp-Davis-Kress) via ASTRA Toolbox or TIGRE (for 3D cone beam)
- Iterative algorithms (SIRT, SART, CGLS) via ASTRA Toolbox or TIGRE

Requires external libraries like ASTRA Toolbox (astra-toolbox) or TIGRE
(tomopy/tigre) for cone-beam and iterative methods.
"""

import numpy as np
import logging
import time
from typing import Optional, Tuple

# Import configuration settings
try:
    import config
except ImportError:
    print("Error: config.py not found. Make sure it's in the src/ directory or Python path.")
    raise

# Configure logging
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Library Import Handling ---
# Attempt to import specialized libraries, providing informative errors if missing.

# ASTRA Toolbox (Example - check actual import name after installation)
try:
    import astra
    ASTRA_AVAILABLE = True
    log.info("ASTRA Toolbox library found.")
except ImportError:
    ASTRA_AVAILABLE = False
    log.warning("ASTRA Toolbox library not found. FDK, SIRT, SART, CGLS (ASTRA) algorithms will not be available.")
    log.warning("Install ASTRA via conda: 'conda install -c astra-toolbox astra-toolbox'")

# TIGRE (Example - often integrated within other packages like TomoPy)
# Adjust import based on how TIGRE is installed/accessed
TIGRE_AVAILABLE = False
# try:
#     import tigre
#     TIGRE_AVAILABLE = True
#     log.info("TIGRE library found.")
# except ImportError:
#     TIGRE_AVAILABLE = False
#     log.warning("TIGRE library not found. FDK, SIRT, SART, CGLS (TIGRE) algorithms will not be available.")
#     log.warning("Install TIGRE following instructions at: https://github.com/CERN/TIGRE")

# scikit-image (for basic FBP)
try:
    from skimage.transform import iradon
    SKIMAGE_AVAILABLE = True
    log.info("scikit-image library found (for basic FBP).")
except ImportError:
    SKIMAGE_AVAILABLE = False
    log.error("scikit-image library not found. Basic FBP algorithm is unavailable.")
    # This is usually a standard dependency, so error might be more appropriate


# --- Helper Functions ---

def get_astra_geometry(cfg: object) -> Optional[Tuple[dict, dict]]:
    """
    Creates ASTRA projection and volume geometry dictionaries from config.

    Args:
        cfg: The configuration object.

    Returns:
        A tuple (proj_geom, vol_geom) containing ASTRA geometry dicts,
        or None if configuration is invalid.
    """
    try:
        if cfg.GEOMETRY_TYPE == 'parallel':
            # Parallel beam geometry
            proj_geom = astra.create_proj_geom(
                'parallel3d',
                cfg.DETECTOR_PIXEL_SIZE_MM, # Detector spacing Y (vertical)
                cfg.DETECTOR_PIXEL_SIZE_MM, # Detector spacing X (horizontal)
                cfg.DETECTOR_SHAPE[0],      # Detector pixel count Y (height/rows)
                cfg.DETECTOR_SHAPE[1],      # Detector pixel count X (width/cols)
                cfg.ANGLES_RAD              # Array of angles in radians
            )
        elif cfg.GEOMETRY_TYPE == 'cone':
             # Cone beam geometry
             # Ensure distances are float
             sod = float(cfg.DISTANCE_SOURCE_OBJECT_MM)
             sdd = float(cfg.DISTANCE_SOURCE_DETECTOR_MM)
             proj_geom = astra.create_proj_geom(
                 'cone',
                 cfg.DETECTOR_PIXEL_SIZE_MM, # Detector spacing Y (vertical)
                 cfg.DETECTOR_PIXEL_SIZE_MM, # Detector spacing X (horizontal)
                 cfg.DETECTOR_SHAPE[0],      # Detector pixel count Y (height/rows)
                 cfg.DETECTOR_SHAPE[1],      # Detector pixel count X (width/cols)
                 cfg.ANGLES_RAD,             # Array of angles in radians
                 sod,                        # Source-Object Distance (SOD)
                 sdd - sod                   # Object-Detector Distance (ODD) = SDD - SOD
             )
        elif cfg.GEOMETRY_TYPE == 'fan':
             log.error("Fan beam geometry setup for ASTRA not implemented in this example.")
             return None
        else:
            log.error(f"Unsupported GEOMETRY_TYPE for ASTRA: {cfg.GEOMETRY_TYPE}")
            return None

        # Volume geometry (defines the reconstruction grid)
        # ASTRA typically expects (X, Y, Z) ordering for volume shape
        vol_geom = astra.create_vol_geom(
            cfg.RECON_VOLUME_SHAPE[2], # Grid size X
            cfg.RECON_VOLUME_SHAPE[1], # Grid size Y
            cfg.RECON_VOLUME_SHAPE[0]  # Grid size Z (number of slices)
        )
        # Optional: Define voxel size if different from detector pixel size mapping
        # vol_geom['option']['voxel_size'] = (vx, vy, vz)

        # Optional: Apply Center of Rotation offset if needed
        # This often requires modifying the projection geometry vectors directly or using specific ASTRA options
        if cfg.CENTER_OF_ROTATION_OFFSET_PX != 0.0:
            log.warning("Center of Rotation offset specified, but applying it in ASTRA requires specific vector adjustments (not implemented in this basic example). Reconstruction may be shifted/blurred.")
            # Example conceptual adjustment (needs verification with ASTRA documentation):
            # offset_mm = cfg.CENTER_OF_ROTATION_OFFSET_PX * cfg.DETECTOR_PIXEL_SIZE_MM
            # proj_geom['option']['ProjectionOrigin'][0] += offset_mm # Adjust projection origin X

        log.info("ASTRA projection geometry created.")
        log.debug(proj_geom)
        log.info("ASTRA volume geometry created.")
        log.debug(vol_geom)
        return proj_geom, vol_geom

    except Exception as e:
        log.error(f"Error creating ASTRA geometry: {e}")
        return None

# --- Reconstruction Algorithms ---

def reconstruct_fbp_skimage(sinogram: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """
    Reconstructs using Filtered Back Projection (FBP) via scikit-image.

    NOTE: skimage.transform.iradon assumes 2D parallel-beam geometry.
    This function applies it slice-by-slice to the 3D sinogram, which is
    mathematically correct only for parallel beam data. It will produce
    significant artifacts for cone-beam or fan-beam data.

    Args:
        sinogram: 3D preprocessed attenuation data (n_proj, height, width).
        cfg: The configuration object.

    Returns:
        Reconstructed 3D volume (height, width, width) or None on error.
    """
    if not SKIMAGE_AVAILABLE:
        log.error("Cannot perform FBP reconstruction: scikit-image is not available.")
        return None

    if cfg.GEOMETRY_TYPE != 'parallel':
        log.warning("Using scikit-image FBP (iradon) for non-parallel beam geometry!")
        log.warning("This assumes slice-independent reconstruction and will likely produce artifacts.")
        log.warning("Consider using ASTRA or TIGRE with FDK/iterative methods for cone/fan beam.")

    num_proj, height, width = sinogram.shape
    angles_deg = cfg.ANGLES_DEG
    filter_name = cfg.FBP_FILTER_NAME
    output_size = cfg.RECON_VOLUME_SHAPE[1] # Assuming square XY reconstruction slices

    log.info(f"Starting scikit-image FBP reconstruction (slice-by-slice)...")
    log.info(f"Parameters: filter='{filter_name}', output_size={output_size}")

    reconstructed_volume = np.zeros((height, output_size, output_size), dtype=np.float32)

    start_time = time.time()
    for i in range(height): # Process each slice (detector row) independently
        slice_sinogram = sinogram[:, i, :] # Shape (n_proj, width)
        log.debug(f"Reconstructing slice {i+1}/{height}...")

        try:
            # iradon expects sinogram shape (width, n_proj) - Transpose needed!
            # Note: circle=True assumes the object is fully within the field of view.
            recon_slice = iradon(
                slice_sinogram.T, # Transpose to (width, n_proj)
                theta=angles_deg,
                output_size=output_size,
                filter_name=filter_name,
                circle=True # Adjust if needed based on object size/scan setup
            )
            reconstructed_volume[i, :, :] = recon_slice.astype(np.float32)
        except Exception as e:
            log.error(f"Error reconstructing slice {i} with skimage.iradon: {e}")
            # Decide whether to continue with zeros or stop
            return None # Stop on first error

    end_time = time.time()
    log.info(f"scikit-image FBP reconstruction finished in {end_time - start_time:.2f} seconds.")
    # Note: Output volume shape is (height, output_size, output_size).
    # Need to align with expected (Z, Y, X) -> (height, output_size, output_size)
    # If config.RECON_VOLUME_SHAPE is (Z,Y,X), this matches if Z=height, Y=X=output_size.
    if reconstructed_volume.shape != cfg.RECON_VOLUME_SHAPE:
         log.warning(f"Output volume shape {reconstructed_volume.shape} differs from config target {cfg.RECON_VOLUME_SHAPE}. Check RECON_VOLUME_SHAPE setting.")

    return reconstructed_volume


def reconstruct_astra(sinogram: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """
    Reconstructs using ASTRA Toolbox algorithms (FDK, SIRT, SART, CGLS).

    Handles geometry setup and algorithm selection based on config.

    Args:
        sinogram: 3D preprocessed attenuation data (n_proj, height, width).
        cfg: The configuration object.

    Returns:
        Reconstructed 3D volume (Z, Y, X) or None on error.
    """
    if not ASTRA_AVAILABLE:
        log.error("Cannot perform ASTRA reconstruction: ASTRA Toolbox library not found.")
        return None

    log.info(f"Starting ASTRA reconstruction using algorithm: {cfg.RECONSTRUCTION_ALGORITHM}")

    # 1. Create ASTRA Geometries
    proj_geom, vol_geom = get_astra_geometry(cfg)
    if proj_geom is None or vol_geom is None:
        log.error("Failed to create ASTRA geometries.")
        return None

    # 2. Prepare Data
    # ASTRA often expects sinogram data in specific order (e.g., (height, n_proj, width))
    # Check ASTRA documentation for the specific algorithm's requirements.
    # Assuming input sinogram is (n_proj, height, width)
    # Let's try transposing to (height, n_proj, width) which is common for ASTRA 3D algos
    try:
        sinogram_astra = np.transpose(sinogram, (1, 0, 2)).copy() # (height, n_proj, width)
        log.info(f"Transposed sinogram for ASTRA to shape: {sinogram_astra.shape}")
    except Exception as e:
        log.error(f"Error transposing sinogram for ASTRA: {e}")
        return None

    # Create sinogram ID in ASTRA
    try:
        sinogram_id = astra.data3d.create('-sino', proj_geom, sinogram_astra)
        log.debug(f"ASTRA sinogram ID created: {sinogram_id}")
    except Exception as e:
        log.error(f"Error creating ASTRA sinogram data object: {e}")
        return None

    # Create volume ID for reconstruction output
    try:
        recon_id = astra.data3d.create('-vol', vol_geom)
        log.debug(f"ASTRA reconstruction volume ID created: {recon_id}")
    except Exception as e:
        log.error(f"Error creating ASTRA volume data object: {e}")
        astra.data3d.delete(sinogram_id) # Clean up previously created object
        return None

    # 3. Configure Reconstruction Algorithm
    algo_name = cfg.RECONSTRUCTION_ALGORITHM.upper() # Ensure uppercase
    astra_cfg = None

    try:
        if algo_name == 'FDK':
            if cfg.GEOMETRY_TYPE != 'cone':
                log.warning(f"FDK algorithm selected but geometry is '{cfg.GEOMETRY_TYPE}'. FDK is for cone-beam.")
            astra_cfg = astra.astra_dict(f'{algo_name}_CUDA') # Use GPU version
            astra_cfg['ProjectionDataId'] = sinogram_id
            astra_cfg['ReconstructionDataId'] = recon_id
            # Optional: Add FDK specific parameters if needed (e.g., filter type, though often built-in)
            # log.info(f"Using FDK filter: (ASTRA default, likely Ramp/Ram-Lak)") # Check ASTRA docs for filter options

        elif algo_name in ['SIRT', 'SART', 'CGLS']:
            astra_cfg = astra.astra_dict(f'{algo_name}_CUDA') # Use GPU version
            astra_cfg['ProjectionDataId'] = sinogram_id
            astra_cfg['ReconstructionDataId'] = recon_id
            # Set common iterative parameters
            astra_cfg['option'] = {}
            astra_cfg['option']['NumIterations'] = cfg.ITERATIVE_NUM_ITERATIONS
            # Add algorithm-specific parameters if needed (e.g., relaxation for SIRT)
            if algo_name == 'SIRT':
                 astra_cfg['option']['Relaxation'] = cfg.ITERATIVE_RELAXATION_PARAM
            # Add constraints (e.g., non-negativity) if desired
            # astra_cfg['option']['MinConstraint'] = 0.0
            log.info(f"Using iterative algorithm {algo_name} with {cfg.ITERATIVE_NUM_ITERATIONS} iterations.")

        else:
            log.error(f"Unsupported ASTRA algorithm specified in config: {cfg.RECONSTRUCTION_ALGORITHM}")
            astra.data3d.delete(sinogram_id, recon_id)
            return None

        # Select GPU if configured and available
        if cfg.USE_GPU:
            try:
                # Find available GPU index (optional, often defaults to 0)
                gpu_index = 0 # Or use astra.gpu.info() or similar to select
                astra_cfg['option']['GPUindex'] = gpu_index
                log.info(f"ASTRA configured to use GPU index: {gpu_index}")
            except Exception as e:
                log.warning(f"Could not set GPU index for ASTRA, might use default. Error: {e}")
        else:
             # Explicitly remove GPUindex if USE_GPU is False, or use CPU versions if available
             log.info("ASTRA configured to use CPU (GPU disabled in config).")
             # Note: ASTRA primarily focuses on GPU. CPU versions might be slower or unavailable for some algos.
             # Change algo name string if CPU versions exist, e.g., 'FDK' instead of 'FDK_CUDA'
             astra_cfg['Algorithm'] = astra_cfg['Algorithm'].replace('_CUDA', '') # Example if CPU versions exist

        log.debug(f"ASTRA algorithm config: {astra_cfg}")

        # 4. Create and Run Algorithm
        alg_id = astra.algorithm.create(astra_cfg)
        log.info(f"Running ASTRA algorithm {algo_name}...")
        start_time = time.time()
        astra.algorithm.run(alg_id)
        end_time = time.time()
        log.info(f"ASTRA algorithm finished in {end_time - start_time:.2f} seconds.")

        # 5. Get Reconstruction Data
        reconstructed_volume = astra.data3d.get(recon_id)
        log.info(f"Reconstructed volume retrieved from ASTRA. Shape: {reconstructed_volume.shape}")

    except Exception as e:
        log.error(f"Error during ASTRA reconstruction process: {e}")
        reconstructed_volume = None
    finally:
        # 6. Clean up ASTRA objects
        astra.algorithm.delete(alg_id if 'alg_id' in locals() else None)
        astra.data3d.delete(sinogram_id if 'sinogram_id' in locals() else None)
        astra.data3d.delete(recon_id if 'recon_id' in locals() else None)
        log.debug("ASTRA objects cleaned up.")

    # Ensure output shape matches config (Z, Y, X)
    # ASTRA output volume is usually (Z, Y, X) if vol_geom was set up correctly
    if reconstructed_volume is not None and reconstructed_volume.shape != cfg.RECON_VOLUME_SHAPE:
         log.warning(f"ASTRA output volume shape {reconstructed_volume.shape} differs from config target {cfg.RECON_VOLUME_SHAPE}. Check volume geometry setup.")

    return reconstructed_volume.astype(np.float32) if reconstructed_volume is not None else None


def reconstruct_tigre(sinogram: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """
    Placeholder for reconstruction using TIGRE library.

    Args:
        sinogram: 3D preprocessed attenuation data (n_proj, height, width).
        cfg: The configuration object.

    Returns:
        Reconstructed 3D volume or None on error.
    """
    if not TIGRE_AVAILABLE:
        log.error("Cannot perform TIGRE reconstruction: TIGRE library not found or enabled.")
        return None

    log.warning("TIGRE reconstruction is not implemented in this example.")
    # --- Implementation Steps ---
    # 1. Define TIGRE geometry object (`tigre.geometry()`) using cfg parameters
    #    (mode='cone', nDetector, dDetector, DSO, DSD, angles, COR offset etc.)
    # 2. Define reconstruction volume parameters (nVoxel, sVoxel)
    # 3. Reshape/transpose sinogram data if needed for TIGRE's expected format.
    # 4. Select TIGRE algorithm function (e.g., `tigre.FDK()`, `tigre.SIRT()`, `tigre.CGLS()`)
    # 5. Call the algorithm function with geometry, data, and parameters (iterations, etc.)
    # 6. Return the reconstructed volume.
    # Example call structure (conceptual):
    # geo = tigre.geometry(mode='cone', ...)
    # nVoxel = np.array(cfg.RECON_VOLUME_SHAPE[::-1]) # TIGRE often uses X,Y,Z
    # sVoxel = np.array([cfg.RECON_VOXEL_SIZE_MM] * 3)
    # if cfg.RECONSTRUCTION_ALGORITHM == 'FDK':
    #     recon = tigre.FDK(sinogram_tigre_format, geo, angles_rad, filter=cfg.FBP_FILTER_NAME)
    # elif cfg.RECONSTRUCTION_ALGORITHM == 'SIRT':
    #     recon = tigre.SIRT(sinogram_tigre_format, geo, angles_rad, niter=cfg.ITERATIVE_NUM_ITERATIONS)
    # ... etc ...
    # return recon

    return None


# --- Main Reconstruction Function ---

def reconstruct_volume(sinogram: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """
    Selects and runs the appropriate reconstruction algorithm based on config.

    Args:
        sinogram: 3D preprocessed attenuation data (n_proj, height, width).
        cfg: The configuration object.

    Returns:
        Reconstructed 3D volume (Z, Y, X) or None if reconstruction fails.
    """
    log.info("--- Starting Reconstruction ---")
    algo = cfg.RECONSTRUCTION_ALGORITHM.upper()
    reconstructed_volume = None

    if algo == 'FBP':
        # Use scikit-image FBP (with warnings for non-parallel beam)
        reconstructed_volume = reconstruct_fbp_skimage(sinogram, cfg)
    elif algo in ['FDK', 'SIRT', 'SART', 'CGLS']:
        # Try ASTRA first if available
        if ASTRA_AVAILABLE:
            reconstructed_volume = reconstruct_astra(sinogram, cfg)
        # Else try TIGRE if available and implemented
        elif TIGRE_AVAILABLE:
             reconstructed_volume = reconstruct_tigre(sinogram, cfg) # Call placeholder
             if reconstructed_volume is None:
                 log.error(f"TIGRE algorithm '{algo}' selected but not implemented or failed.")
        else:
            log.error(f"Algorithm '{algo}' requires ASTRA Toolbox or TIGRE, but neither is available/enabled.")
    else:
        log.error(f"Unknown reconstruction algorithm specified in config: {cfg.RECONSTRUCTION_ALGORITHM}")

    if reconstructed_volume is not None:
        log.info(f"--- Reconstruction Finished Successfully (Algorithm: {algo}) ---")
        log.info(f"Final reconstructed volume shape: {reconstructed_volume.shape}, dtype: {reconstructed_volume.dtype}")
    else:
        log.error(f"--- Reconstruction Failed (Algorithm: {algo}) ---")

    return reconstructed_volume


# Example usage (for testing purposes)
if __name__ == '__main__':
    import os
    import imageio.v3 as iio # To save sample slice

    print("--- Running Reconstruction Test ---")

    # Ensure output directories exist for logging and saving plots
    config.ensure_output_dirs_exist()

    # Configure logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='a') # Append to log
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)
    log.addHandler(logging.StreamHandler())
    log.propagate = False

    log.info("Starting reconstruction test script.")

    # --- Create Dummy Sinogram Data ---
    # In a real run, this would be loaded from the preprocessor output
    log.warning("Using dummy sinogram data for reconstruction test.")
    n_proj = config.NUM_PROJECTIONS
    height = config.DETECTOR_SHAPE[0] # Z dimension (slices)
    width = config.DETECTOR_SHAPE[1]  # X dimension (detector pixels)
    # Create a simple sinogram (e.g., a cylinder) - this is just for testing flow
    # A real sinogram would come from preprocess_data()
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    dummy_sinogram = np.zeros((n_proj, height, width), dtype=np.float32)
    # Simple simulation (replace with loading actual preprocessed data)
    for i, angle in enumerate(config.ANGLES_RAD):
        # This simulation is extremely basic and not physically accurate
        # It just creates *some* non-zero data structure
        offset = int(radius * np.cos(angle + np.pi/2))
        col_start = max(0, center_x - radius // 4 + offset)
        col_end = min(width, center_x + radius // 4 + offset)
        row_start = max(0, center_y - radius)
        row_end = min(height, center_y + radius)
        if col_start < col_end and row_start < row_end:
             dummy_sinogram[i, row_start:row_end, col_start:col_end] = 0.01 # Small attenuation value

    log.info(f"Created dummy sinogram with shape: {dummy_sinogram.shape}")

    # --- Run Reconstruction ---
    recon_volume = reconstruct_volume(dummy_sinogram, config)

    # --- Process Results ---
    if recon_volume is not None:
        log.info(f"Reconstruction test successful. Volume shape: {recon_volume.shape}")
        log.info(f"Volume min/max values: {np.min(recon_volume):.4f} / {np.max(recon_volume):.4f}")

        # Optional: Save a central slice for visual inspection
        try:
            slice_idx = recon_volume.shape[0] // 2 # Middle Z slice
            recon_slice = recon_volume[slice_idx, :, :]
            recon_slice_path = os.path.join(config.RECON_SLICES_DIR, f'test_recon_slice_{config.RECONSTRUCTION_ALGORITHM}.png')

            # Normalize slice for saving as image
            recon_slice_norm = (recon_slice - np.min(recon_slice)) / (np.max(recon_slice) - np.min(recon_slice) + EPSILON)
            recon_slice_uint8 = (recon_slice_norm * 255).astype(np.uint8)

            iio.imwrite(recon_slice_path, recon_slice_uint8, prefer_uint8=True) # Specify uint8 preference
            log.info(f"Saved sample reconstructed slice to {recon_slice_path}")

        except ImportError:
            log.warning("imageio not found. Cannot save sample reconstruction slice.")
        except Exception as e:
            log.error(f"Error saving sample reconstruction slice: {e}")
    else:
        log.error("Reconstruction test failed.")

    log.info("Reconstruction test script finished.")
    print("--- End of Reconstruction Test ---")
    print(f"Check log file for details: {config.LOG_FILE}")

