"""
ASTRA Toolbox-based reconstruction algorithms for tomographic reconstruction.
This provides high-performance GPU-accelerated implementations using the ASTRA Toolbox.
"""
import numpy as np

try:
    import astra
    ASTRA_AVAILABLE = True
    print("ASTRA Toolbox is available. GPU-accelerated reconstruction enabled.")
except ImportError:
    ASTRA_AVAILABLE = False
    print("ASTRA Toolbox not available. Install with 'conda install -c astra-toolbox astra-toolbox'")

def filtered_backprojection_astra(projections, angles, volume_shape=None, filter_name='ram-lak', downsample_factor=1):
    """
    ASTRA-based Filtered backprojection algorithm.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple, optional): Shape of the output volume (height, width, slices).
        filter_name (str): Filter to use (ram-lak, shepp-logan, cosine, hamming, hann, etc.).
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not ASTRA_AVAILABLE:
        raise ImportError("ASTRA Toolbox is required for this function. "
                         "Install with: conda install -c astra-toolbox astra-toolbox")
    
    # Handle angle mismatch - this sometimes happens with real datasets
    if projections.shape[0] != len(angles):
        print(f"WARNING: Projection count ({projections.shape[0]}) doesn't match angles ({len(angles)})")
        # Use the minimum count
        count = min(projections.shape[0], len(angles))
        projections = projections[:count]
        angles = angles[:count]
    
    # Apply downsampling if requested
    if downsample_factor > 1:
        print(f"Downsampling projections by factor {downsample_factor}")
        # Downsample detector rows/columns - skip pixels
        projections_ds = projections[:, ::downsample_factor, ::downsample_factor]
        print(f"Downsampled projection shape: {projections_ds.shape} (original: {projections.shape})")
        projections = projections_ds
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)
    
    if volume_shape is None:
        size = projections.shape[2]
        volume_shape = (size, size, projections.shape[1])
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    print(f"ASTRA FBP: Processing {projections.shape[1]} slices with {len(angles)} projection angles")
    
    # Process slices
    for slice_idx in range(projections.shape[1]):
        # Extract sinogram for this slice
        sino = projections[:, slice_idx, :]
        
        # Print dimensions for debugging
        print(f"Sinogram shape: {sino.shape}")
        det_count = sino.shape[1]
        print(f"Creating geometry with {det_count} detectors and {len(angles)} angles")
        
        #----------------------------------------------------------------------
        # ASTRA DIRECT IMPLEMENTATION
        #----------------------------------------------------------------------
        try:
            # Step 1: Create volume geometry
            vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1])
            
            # Step 2: Create sinogram geometry - THIS IS THE KEY!
            # Note: In ASTRA, detector count is the WIDTH of the detector
            proj_geom = astra.create_proj_geom('parallel', 1.0, det_count, angles_rad)
                
            # Step 3: Create a sinogram data object explicitly
            # This is the critical part: ASTRA expects sinogram as (angles, detectors)
            sino_id = astra.data2d.create('-sino', proj_geom, data=sino)
            
            # Step 4: Create reconstruction data object
            vol_id = astra.data2d.create('-vol', vol_geom)
            
            # Step 5: Setup FBP algorithm
            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            cfg['FilterType'] = filter_name
            
            # Step 6: Run the algorithm
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            
            # Step 7: Get result
            reconstruction = astra.data2d.get(vol_id)
            volume[:, :, slice_idx] = reconstruction
            
            # Step 8: Clean up memory
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)
            
        except Exception as e:
            print(f"ASTRA FBP failed with error: {e}")
            raise
    
    return volume

def sirt_reconstruction_astra(projections, angles, volume_shape, iterations=100, downsample_factor=1):
    """
    ASTRA-based SIRT reconstruction algorithm.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple): Shape of the output volume (height, width, slices).
        iterations (int): Number of iterations.
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not ASTRA_AVAILABLE:
        raise ImportError("ASTRA Toolbox is required for this function. "
                         "Install with: conda install -c astra-toolbox astra-toolbox")
    
    # Handle angle mismatch
    if projections.shape[0] != len(angles):
        print(f"WARNING: Projection count ({projections.shape[0]}) doesn't match angles ({len(angles)})")
        # Use the minimum count
        count = min(projections.shape[0], len(angles))
        projections = projections[:count]
        angles = angles[:count]
        
    # Apply downsampling if requested
    if downsample_factor > 1:
        print(f"Downsampling projections by factor {downsample_factor}")
        # Downsample detector rows/columns - skip pixels
        projections_ds = projections[:, ::downsample_factor, ::downsample_factor]
        print(f"Downsampled projection shape: {projections_ds.shape} (original: {projections.shape})")
        projections = projections_ds
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # Calculate the number of slices to process (don't exceed volume depth)
    slices_to_process = min(projections.shape[1], volume_shape[2])
    
    print(f"ASTRA SIRT: Processing {slices_to_process} slices with {len(angles)} projection angles")
    print(f"Volume shape: {volume_shape}, Projections shape: {projections.shape}")
    
    # Process slices - only up to the minimum of projection slices or volume depth
    for slice_idx in range(slices_to_process):
        # Extract sinogram for this slice
        sino = projections[:, slice_idx, :]
        
        # Print dimensions for debugging
        print(f"Sinogram shape: {sino.shape}")
        det_count = sino.shape[1]
        print(f"Creating geometry with {det_count} detectors and {len(angles)} angles")
        
        try:
            # Create volume geometry
            vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1])
            
            # Create sinogram geometry
            proj_geom = astra.create_proj_geom('parallel', 1.0, det_count, angles_rad)
                
            # Create a sinogram data object - ASTRA expects (angles, detectors)
            sino_id = astra.data2d.create('-sino', proj_geom, data=sino)
            
            # Create reconstruction data object
            vol_id = astra.data2d.create('-vol', vol_geom)
            
            # Setup SIRT algorithm
            cfg = astra.astra_dict('SIRT_CUDA')
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            
            # Run the algorithm
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, iterations)
            
            # Get result
            reconstruction = astra.data2d.get(vol_id)
            
            # Make sure we're not accessing out of bounds
            if slice_idx < volume_shape[2]:
                volume[:, :, slice_idx] = reconstruction
            else:
                print(f"WARNING: Slice index {slice_idx} exceeds volume depth {volume_shape[2]}")
                break  # Stop processing if we've reached the volume limit
            
            # Clean up memory
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)
            
        except Exception as e:
            print(f"ASTRA SIRT failed with error: {e}")
            raise
    
    return volume

def fdk_reconstruction_astra(projections, geometry, volume_shape):
    """
    ASTRA-based FDK reconstruction algorithm for cone-beam geometry.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        geometry (dict): Projection geometry with source_origin_dist and origin_detector_dist.
        volume_shape (tuple): Shape of the output volume (height, width, depth).
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    if not ASTRA_AVAILABLE:
        raise ImportError("ASTRA Toolbox is required for this function. "
                         "Install with: conda install -c astra-toolbox astra-toolbox")
    
    # Extract geometry parameters
    angles_deg = geometry['angles']
    
    # Handle angle mismatch
    if projections.shape[0] != len(angles_deg):
        print(f"WARNING: Projection count ({projections.shape[0]}) doesn't match angles ({len(angles_deg)})")
        # Use the minimum count
        count = min(projections.shape[0], len(angles_deg))
        projections = projections[:count]
        angles_deg = angles_deg[:count]
    
    # Extract remaining geometry parameters
    source_origin_dist = float(geometry.get('source_origin_dist', 500.0))
    origin_detector_dist = float(geometry.get('origin_detector_dist', 500.0))
    detector_width = projections.shape[2]
    detector_height = projections.shape[1]
    
    print(f"FDK Geometry: {len(angles_deg)} angles, detector size {detector_width}x{detector_height}")
    print(f"Source-Origin: {source_origin_dist}mm, Origin-Detector: {origin_detector_dist}mm")
    
    # Convert to radians
    angles_rad = np.deg2rad(angles_deg)
    
    try:
        # Create 3D volume geometry
        vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1], volume_shape[2])
        
        # Create vectors for cone beam geometry
        vectors = np.zeros((len(angles_rad), 12))
        for i, angle in enumerate(angles_rad):
            # Source position
            vectors[i, 0] = np.sin(angle) * source_origin_dist  # x
            vectors[i, 1] = -np.cos(angle) * source_origin_dist  # y
            vectors[i, 2] = 0  # z
            
            # Detector center
            vectors[i, 3] = -np.sin(angle) * origin_detector_dist  # x
            vectors[i, 4] = np.cos(angle) * origin_detector_dist  # y
            vectors[i, 5] = 0  # z
            
            # Detector u direction (columns)
            vectors[i, 6] = np.cos(angle)  # x
            vectors[i, 7] = np.sin(angle)  # y
            vectors[i, 8] = 0  # z
            
            # Detector v direction (rows)
            vectors[i, 9] = 0  # x
            vectors[i, 10] = 0  # y
            vectors[i, 11] = 1  # z
        
        # Create projection geometry
        proj_geom = astra.create_proj_geom('cone_vec', detector_height, detector_width, vectors)
        
        # Create projection data
        # ASTRA requires dimensions: (detector_rows, projection_count, detector_cols)
        projs_astra = np.transpose(projections, (1, 0, 2))
        proj_id = astra.data3d.create('-proj3d', proj_geom, projs_astra)
        
        # Create volume data
        vol_id = astra.data3d.create('-vol', vol_geom)
        
        # Setup FDK reconstruction
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectionDataId'] = proj_id
        
        # Run the algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # Get reconstruction
        volume = astra.data3d.get(vol_id)
        
        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)
        
    except Exception as e:
        print(f"ASTRA FDK failed with error: {e}")
        raise
    
    return volume

def create_astra_geometric_parameters(geometry_dict):
    """
    Create ASTRA-compatible geometric parameters from our geometry dictionary.
    
    Args:
        geometry_dict (dict): Our geometry dictionary.
        
    Returns:
        dict: ASTRA-compatible geometry parameters.
    """
    # Extract parameters
    angles = geometry_dict['angles']
    detector_shape = geometry_dict['detector_shape']
    source_origin_dist = geometry_dict['source_origin_dist']
    origin_detector_dist = geometry_dict['origin_detector_dist']
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)
    
    # Create parameter dictionary
    astra_params = {
        'angles': angles_rad,
        'detector_shape': detector_shape,
        'source_origin_dist': source_origin_dist,
        'origin_detector_dist': origin_detector_dist
    }
    
    return astra_params
