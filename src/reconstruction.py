# -*- coding: utf-8 -*-
"""
Module for performing CT reconstruction.
(Modified reconstruct_astra for Z-axis chunking to reduce memory usage).
"""

import numpy as np
import logging
import time
import gc # Import garbage collector
from typing import Optional, Tuple, Dict

# Import configuration settings
try:
    import config
    EPSILON = getattr(config, 'EPSILON', 1e-9)
except ImportError:
    print("Error: config.py not found. Make sure it's in the src/ directory or Python path.")
    EPSILON = 1e-9 # Default epsilon if config fails
    raise

# Configure logging
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=getattr(config, config.LOG_LEVEL, logging.INFO),
                        format='%(asctime)s [%(levelname)-5.5s] %(name)-15s - %(message)s')

# --- Library Import Handling ---
# ASTRA Toolbox
try:
    import astra
    ASTRA_AVAILABLE = True
    log.info(f"ASTRA Toolbox library found. Version: {getattr(astra, '__version__', 'Unknown')}")
except ImportError:
    ASTRA_AVAILABLE = False
    log.warning("ASTRA Toolbox library not found. FDK, SIRT, SART, CGLS (ASTRA) algorithms will not be available.")

# TIGRE (Placeholder)
TIGRE_AVAILABLE = False

# scikit-image (for basic FBP)
try:
    from skimage.transform import iradon
    SKIMAGE_AVAILABLE = True
    log.info("scikit-image library found (for basic FBP).")
except ImportError:
    SKIMAGE_AVAILABLE = False
    log.error("scikit-image library not found. Basic FBP algorithm is unavailable.")


# --- Helper Functions ---
# get_astra_geometry remains the same as the previous corrected version
def get_astra_geometry(cfg: object) -> Optional[Tuple[dict, dict]]:
    """
    Creates ASTRA projection and volume geometry dictionaries from config.
    (Uses full volume geometry for reference, chunking handled in reconstruct_astra).
    """
    log.debug("Attempting to create ASTRA geometry...")
    if not ASTRA_AVAILABLE:
        log.error("ASTRA library not available for geometry creation.")
        return None
    try:
        # --- Ensure scalar types for arguments ---
        try:
            det_height = int(cfg.DETECTOR_SHAPE[0])
            det_width = int(cfg.DETECTOR_SHAPE[1])
            pixel_size = float(cfg.DETECTOR_PIXEL_SIZE_MM)
            sod = float(cfg.DISTANCE_SOURCE_OBJECT_MM)
            sdd = float(cfg.DISTANCE_SOURCE_DETECTOR_MM)
            odd = sdd - sod
            angles_rad = cfg.ANGLES_RAD
            if not isinstance(angles_rad, np.ndarray):
                 angles_rad = np.array(angles_rad, dtype=float)
            if sod <= 0 or odd <= 0 or sdd <= 0:
                 log.error(f"Invalid distances: SOD={sod}, SDD={sdd}, ODD={odd}. Must be positive.")
                 return None
        except Exception as e:
            log.error(f"Error converting config parameters to required types: {e}", exc_info=True)
            return None

        # --- Create Projection Geometry (Always for full detector) ---
        proj_geom = None
        if cfg.GEOMETRY_TYPE == 'parallel':
            proj_geom = astra.create_proj_geom('parallel3d', pixel_size, pixel_size, det_height, det_width, angles_rad)
        elif cfg.GEOMETRY_TYPE == 'cone':
             proj_geom = astra.create_proj_geom('cone', pixel_size, pixel_size, det_height, det_width, angles_rad, sod, odd)
        else:
            log.error(f"Unsupported GEOMETRY_TYPE for ASTRA: {cfg.GEOMETRY_TYPE}")
            return None
        if proj_geom is None or not isinstance(proj_geom, dict):
             log.error(f"ASTRA create_proj_geom failed.")
             return None
        log.info("ASTRA projection geometry created successfully.")

        # --- Create Volume Geometry (For the *full* target volume shape) ---
        # Chunking logic will create smaller geometries based on this later
        try:
             vol_shape_x = int(cfg.RECON_VOLUME_SHAPE[2])
             vol_shape_y = int(cfg.RECON_VOLUME_SHAPE[1])
             vol_shape_z = int(cfg.RECON_VOLUME_SHAPE[0])
             if not (vol_shape_x > 0 and vol_shape_y > 0 and vol_shape_z > 0):
                  log.error(f"Invalid RECON_VOLUME_SHAPE: {cfg.RECON_VOLUME_SHAPE}.")
                  return None
        except Exception as e:
             log.error(f"Error converting RECON_VOLUME_SHAPE {cfg.RECON_VOLUME_SHAPE} elements to int: {e}")
             return None
        vol_geom = astra.create_vol_geom(vol_shape_x, vol_shape_y, vol_shape_z)
        if vol_geom is None or not isinstance(vol_geom, dict):
             log.error(f"ASTRA create_vol_geom failed.")
             return None
        log.info("ASTRA volume geometry (full) created successfully.")

        # Optional: Apply Center of Rotation offset to proj_geom
        if hasattr(cfg, 'CENTER_OF_ROTATION_OFFSET_PX') and cfg.CENTER_OF_ROTATION_OFFSET_PX != 0.0:
            offset_px = float(cfg.CENTER_OF_ROTATION_OFFSET_PX)
            log.warning(f"Center of Rotation offset specified ({offset_px} px), attempting basic adjustment.")
            try:
                offset_mm = offset_px * pixel_size
                if 'DetectorOrigin' in proj_geom:
                     current_origin = list(proj_geom['DetectorOrigin'])
                     current_origin[0] += offset_mm
                     proj_geom['DetectorOrigin'] = tuple(current_origin)
                     log.info(f"Adjusted proj_geom['DetectorOrigin'] by {offset_mm:.4f} mm for COR offset.")
                else:
                     log.warning("Could not find 'DetectorOrigin' key in proj_geom. COR offset not applied via this method.")
            except Exception as e:
                 log.error(f"Failed to apply COR offset to proj_geom: {e}")

        return proj_geom, vol_geom
    except Exception as e:
        log.error(f"Unhandled error creating ASTRA geometry: {e}", exc_info=True)
        return None

# --- Reconstruction Algorithms ---
# reconstruct_fbp_skimage remains the same
def reconstruct_fbp_skimage(sinogram: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """ FBP reconstruction using scikit-image (slice-by-slice). """
    # ... (implementation from previous version) ...
    if not SKIMAGE_AVAILABLE: return None
    if cfg.GEOMETRY_TYPE != 'parallel': log.warning("Using skimage FBP for non-parallel beam!")
    num_proj, height, width = sinogram.shape
    angles_deg = cfg.ANGLES_DEG
    filter_name = cfg.FBP_FILTER_NAME
    output_size = cfg.RECON_VOLUME_SHAPE[1]
    log.info(f"Starting scikit-image FBP reconstruction...")
    reconstructed_volume = np.zeros((height, output_size, output_size), dtype=np.float32)
    start_time = time.time()
    for i in range(height):
        slice_sinogram = sinogram[:, i, :]
        log.debug(f"Reconstructing slice {i+1}/{height}...")
        try:
            recon_slice = iradon(slice_sinogram.T, theta=angles_deg, output_size=output_size, filter_name=filter_name, circle=True)
            reconstructed_volume[i, :, :] = recon_slice.astype(np.float32)
        except Exception as e: log.error(f"Error reconstructing slice {i}: {e}"); return None
    end_time = time.time(); log.info(f"scikit-image FBP finished in {end_time - start_time:.2f}s.")
    if reconstructed_volume.shape != cfg.RECON_VOLUME_SHAPE: log.warning(f"Output shape {reconstructed_volume.shape} differs from target {cfg.RECON_VOLUME_SHAPE}.")
    return reconstructed_volume


# --- MODIFIED reconstruct_astra function ---
def reconstruct_astra(sinogram: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """
    Reconstructs using ASTRA Toolbox algorithms (FDK, SIRT, SART, CGLS).
    Includes Z-axis chunking for memory optimization.

    Args:
        sinogram: 3D preprocessed attenuation data (n_proj, height, width).
                  Expected to be a standard NumPy array.
        cfg: The configuration object.

    Returns:
        Reconstructed 3D volume (Z, Y, X) or None on error.
    """
    if not ASTRA_AVAILABLE:
        log.error("Cannot perform ASTRA reconstruction: ASTRA Toolbox library not found.")
        return None
    if not isinstance(sinogram, np.ndarray):
        log.error(f"Input sinogram is not a NumPy array (type: {type(sinogram)}). Cannot proceed.")
        return None

    log.info(f"Starting ASTRA reconstruction using algorithm: {cfg.RECONSTRUCTION_ALGORITHM} with Z-chunking.")

    # --- Get Overall Geometry ---
    log.info("Creating full ASTRA geometries for reference...")
    geom_result = get_astra_geometry(cfg)
    if geom_result is None: return None
    proj_geom, full_vol_geom = geom_result
    if proj_geom is None or full_vol_geom is None: return None

    # --- Prepare Full Sinogram Data ---
    log.info("Preparing full sinogram data for ASTRA...")
    sinogram_id = None # Initialize for cleanup
    try:
        sinogram_transposed = np.transpose(sinogram, (1, 0, 2))
        sinogram_astra = np.ascontiguousarray(sinogram_transposed, dtype=np.float32)
        del sinogram_transposed # Free memory if copy was made
        gc.collect() # Encourage garbage collection
        log.info(f"Prepared sinogram. Shape: {sinogram_astra.shape}, Dtype: {sinogram_astra.dtype}")

        # Create ASTRA object for the *full* sinogram
        sinogram_id = astra.data3d.create('-sino', proj_geom, sinogram_astra)
        log.info(f"ASTRA sinogram ID created: {sinogram_id}")
        # We can potentially free the NumPy array now if ASTRA copies it internally
        # Be cautious with this - test if it causes issues
        del sinogram_astra
        gc.collect()
        log.debug("Released NumPy sinogram reference after passing to ASTRA.")

    except Exception as e:
        log.error(f"Error preparing sinogram or creating ASTRA sinogram object: {e}", exc_info=True)
        if sinogram_id is not None: astra.data3d.delete(sinogram_id)
        return None

    # --- Setup Chunking ---
    full_z, full_y, full_x = cfg.RECON_VOLUME_SHAPE
    chunk_size = getattr(cfg, 'RECON_Z_CHUNK_SIZE', None)

    # Disable chunking if chunk_size is None, 0, or >= full_z
    if chunk_size is None or chunk_size <= 0 or chunk_size >= full_z:
        chunk_size = full_z
        log.info("Chunking disabled or chunk size covers full volume. Processing as single chunk.")
    else:
        chunk_size = int(chunk_size)
        log.info(f"Processing reconstruction in Z-chunks of size: {chunk_size}")

    num_chunks = (full_z + chunk_size - 1) // chunk_size # Ceiling division
    log.info(f"Total Z-slices: {full_z}, Number of chunks: {num_chunks}")

    # Pre-allocate final volume in NumPy
    final_reconstructed_volume = np.zeros(cfg.RECON_VOLUME_SHAPE, dtype=np.float32)
    total_recon_time = 0

    # --- Loop Through Chunks ---
    for i in range(num_chunks):
        chunk_start_z = i * chunk_size
        chunk_end_z = min((i + 1) * chunk_size, full_z)
        current_chunk_z = chunk_end_z - chunk_start_z
        log.info(f"--- Processing Chunk {i+1}/{num_chunks} (Z-slices: {chunk_start_z} to {chunk_end_z-1}) ---")

        alg_id = None
        recon_id = None
        chunk_vol_geom = None

        try:
            # 1. Create Volume Geometry for the *current chunk*
            chunk_vol_geom = astra.create_vol_geom(full_x, full_y, current_chunk_z)
            if chunk_vol_geom is None: raise ValueError("Failed to create chunk volume geometry.")
            log.debug(f"Created chunk volume geometry: {chunk_vol_geom}")

            # 2. Create ASTRA data object for the chunk volume
            recon_id = astra.data3d.create('-vol', chunk_vol_geom, 0)
            if recon_id is None: raise ValueError("Failed to create chunk volume data ID.")
            log.debug(f"Created chunk reconstruction data ID: {recon_id}")

            # 3. Configure Algorithm for this chunk
            algo_name = cfg.RECONSTRUCTION_ALGORITHM.upper()
            use_gpu_algo = cfg.USE_GPU
            # (GPU check logic - simplified here, assumes check done once outside loop)
            algo_string = f'{algo_name}_CUDA' if use_gpu_algo else algo_name

            astra_cfg = astra.astra_dict(algo_string)
            if astra_cfg is None: raise ValueError(f"astra.astra_dict returned None for '{algo_string}'.")

            astra_cfg['ProjectionDataId'] = sinogram_id # Use full sinogram
            astra_cfg['ReconstructionDataId'] = recon_id # Use chunked volume

            # Add algorithm-specific options (same as before)
            if algo_name == 'FDK': pass # No extra options needed usually
            elif algo_name in ['SIRT', 'SART', 'CGLS']:
                astra_cfg['option'] = {}
                astra_cfg['option']['NumIterations'] = int(cfg.ITERATIVE_NUM_ITERATIONS)
                if algo_name == 'SIRT': astra_cfg['option']['Relaxation'] = float(cfg.ITERATIVE_RELAXATION_PARAM)
            if use_gpu_algo:
                astra_cfg['option'] = astra_cfg.get('option', {})
                astra_cfg['option']['GPUindex'] = 0

            log.debug(f"Algorithm config for chunk {i+1}: {astra_cfg}")

            # 4. Create and Run Algorithm for the chunk
            alg_id = astra.algorithm.create(astra_cfg)
            if alg_id is None: raise ValueError("Failed to create algorithm object.")
            log.info(f"Running ASTRA algorithm for chunk {i+1}...")
            chunk_start_time = time.time()
            astra.algorithm.run(alg_id)
            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time
            total_recon_time += chunk_duration
            log.info(f"Chunk {i+1} finished in {chunk_duration:.2f} seconds.")

            # 5. Get Reconstructed Chunk Data
            chunk_data = astra.data3d.get(recon_id)
            if chunk_data is None: raise ValueError("Failed to retrieve reconstructed chunk data.")
            log.debug(f"Retrieved chunk data shape: {chunk_data.shape}")

            # 6. Place chunk into the final volume array
            if chunk_data.shape == (current_chunk_z, full_y, full_x):
                 final_reconstructed_volume[chunk_start_z:chunk_end_z, :, :] = chunk_data
                 log.debug(f"Stored chunk {i+1} into final volume.")
            else:
                 log.error(f"Shape mismatch for chunk {i+1}! Expected {(current_chunk_z, full_y, full_x)}, got {chunk_data.shape}. Skipping storage.")
                 # Consider stopping if shape mismatch occurs

        except Exception as e:
            log.error(f"Error processing chunk {i+1} (Z={chunk_start_z}-{chunk_end_z-1}): {e}", exc_info=True)
            # Cleanup sinogram_id and return None if a chunk fails critically
            if sinogram_id is not None: astra.data3d.delete(sinogram_id)
            # Try to clean up current chunk objects as well
            if alg_id is not None: astra.algorithm.delete(alg_id)
            if recon_id is not None: astra.data3d.delete(recon_id)
            return None # Indicate failure
        finally:
            # 7. Clean up ASTRA objects for *this chunk* within the loop
            log.debug(f"Cleaning up ASTRA objects for chunk {i+1}...")
            if alg_id is not None: astra.algorithm.delete(alg_id)
            if recon_id is not None: astra.data3d.delete(recon_id)
            # We do NOT delete sinogram_id here, only after the loop
            gc.collect() # Encourage memory release

    # --- End of Chunk Loop ---

    # Clean up the full sinogram object *after* all chunks are processed
    log.debug("Cleaning up full sinogram ASTRA object...")
    if sinogram_id is not None: astra.data3d.delete(sinogram_id)
    gc.collect()

    log.info(f"Total ASTRA reconstruction time across all chunks: {total_recon_time:.2f} seconds.")

    # Final check on output shape
    if final_reconstructed_volume.shape != cfg.RECON_VOLUME_SHAPE:
         log.warning(f"Final assembled volume shape {final_reconstructed_volume.shape} differs from config target {cfg.RECON_VOLUME_SHAPE}.")

    return final_reconstructed_volume # Already float32

# reconstruct_tigre remains the same placeholder
def reconstruct_tigre(sinogram: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """ Placeholder for reconstruction using TIGRE library. """
    if not TIGRE_AVAILABLE: log.error("TIGRE library not found."); return None
    log.warning("TIGRE reconstruction is not implemented."); return None

# reconstruct_volume remains the same dispatcher
def reconstruct_volume(sinogram: np.ndarray, cfg: object) -> Optional[np.ndarray]:
    """ Selects and runs the appropriate reconstruction algorithm based on config. """
    log.info("--- Starting Reconstruction ---")
    algo = cfg.RECONSTRUCTION_ALGORITHM.upper()
    reconstructed_volume = None
    if algo == 'FBP':
        reconstructed_volume = reconstruct_fbp_skimage(sinogram, cfg)
    elif algo in ['FDK', 'SIRT', 'SART', 'CGLS']:
        if ASTRA_AVAILABLE:
            reconstructed_volume = reconstruct_astra(sinogram, cfg) # Calls the chunked version
        elif TIGRE_AVAILABLE:
             reconstructed_volume = reconstruct_tigre(sinogram, cfg)
             if reconstructed_volume is None: log.error(f"TIGRE algorithm '{algo}' failed.")
        else:
            log.error(f"Algorithm '{algo}' requires ASTRA or TIGRE, but neither is available.")
    else:
        log.error(f"Unknown reconstruction algorithm: {cfg.RECONSTRUCTION_ALGORITHM}")

    if reconstructed_volume is not None:
        log.info(f"--- Reconstruction Finished Successfully (Algorithm: {algo}) ---")
    else:
        log.error(f"--- Reconstruction Failed (Algorithm: {algo}) ---")
    return reconstructed_volume


# Example usage (for testing purposes)
if __name__ == '__main__':
    # ... (test script remains similar, will now use chunking if configured) ...
    import os, imageio.v3 as iio
    print("--- Running Reconstruction Test (with chunking logic) ---")
    config.ensure_output_dirs_exist()
    # ... (logging setup) ...
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    log_file_handler = logging.FileHandler(config.LOG_FILE, mode='a')
    log_file_handler.setFormatter(log_formatter)
    log.addHandler(log_file_handler)
    log.addHandler(logging.StreamHandler())
    log.propagate = False
    log.info("Starting reconstruction test script (chunking enabled if configured).")
    log.warning("Using dummy sinogram data for reconstruction test.")
    n_proj = config.NUM_PROJECTIONS
    try: height, width = config.DETECTOR_SHAPE
    except: height, width = 512, 512
    dummy_sinogram = np.zeros((n_proj, height, width), dtype=np.float32)
    log.info(f"Created dummy sinogram with shape: {dummy_sinogram.shape}")
    recon_volume = reconstruct_volume(dummy_sinogram, config)
    if recon_volume is not None:
        log.info(f"Reconstruction test successful. Volume shape: {recon_volume.shape}")
        # ... (save sample slice logic) ...
        try:
            slice_idx = recon_volume.shape[0] // 2
            recon_slice = recon_volume[slice_idx, :, :]
            recon_slice_path = os.path.join(config.RECON_SLICES_DIR, f'test_recon_slice_{config.RECONSTRUCTION_ALGORITHM}.png')
            recon_slice_norm = (recon_slice - np.min(recon_slice)) / (np.max(recon_slice) - np.min(recon_slice) + EPSILON)
            recon_slice_uint8 = (np.clip(recon_slice_norm,0,1) * 255).astype(np.uint8)
            iio.imwrite(recon_slice_path, recon_slice_uint8, prefer_uint8=True)
            log.info(f"Saved sample reconstructed slice to {recon_slice_path}")
        except Exception as e: log.error(f"Error saving sample reconstruction slice: {e}")
    else: log.error("Reconstruction test failed.")
    log.info("Reconstruction test script finished.")
    print("--- End of Reconstruction Test ---")
