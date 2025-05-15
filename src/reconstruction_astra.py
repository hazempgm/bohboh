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

def filtered_backprojection_astra(projections, angles, volume_shape=None, filter_name='ram-lak'):
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
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)
    
    if volume_shape is None:
        size = projections.shape[2]  # Use width as size
        volume_shape = (size, size, projections.shape[1])
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    print(f"ASTRA: Processing {projections.shape[1]} slices with {len(angles)} projection angles")
    
    # Handle angle count mismatches gracefully
    if projections.shape[0] != len(angles):
        print(f"WARNING: Projection count ({projections.shape[0]}) doesn't match angles ({len(angles)})")
        # Fix the mismatch by using the smaller count
        if projections.shape[0] > len(angles):
            print(f"Trimming projections from {projections.shape[0]} to {len(angles)}")
            projections = projections[:len(angles)]
        else:
            print(f"Trimming angles from {len(angles)} to {projections.shape[0]}")
            angles = angles[:projections.shape[0]]
            angles_rad = np.deg2rad(angles)
    
    # ASTRA 2D FBP for parallel beam works on a slice-by-slice basis
    try:
        # Process each slice
        for slice_idx in range(projections.shape[1]):
            # Extract sinogram for current slice - shape: (angles, width)
            sinogram = projections[:, slice_idx, :]
            
            # Create ASTRA volume geometry
            vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1])
            
            # COMPLETELY DIFFERENT APPROACH:
            # We'll do a direct raw data copy to work around ASTRA's dimension expectations
            
            # Create parallel projection geometry with vectors
            det_count = sinogram.shape[1]
            angle_count = sinogram.shape[0]
            
            # Log geometry info for debugging
            if slice_idx == 0:
                print(f"Creating geometry with {det_count} detectors and {angle_count} angles")
                print(f"Sinogram shape: {sinogram.shape}")
            
            # Create custom projection vectors
            vectors = np.zeros((angle_count, 6))
            for i, angle in enumerate(angles_rad):
                # Ray direction
                vectors[i, 0] = np.cos(angle) 
                vectors[i, 1] = np.sin(angle)
                # Center of detector
                vectors[i, 2] = 0
                vectors[i, 3] = 0  
                # Vector from detector pixel 0 to 1
                vectors[i, 4] = -np.sin(angle)
                vectors[i, 5] = np.cos(angle)
                
            # Create vector-based projection geometry (more flexible)
            proj_geom = astra.create_proj_geom('parallel_vec', det_count, vectors)
            
            # Create a temporary memory block for the sinogram (rotated correctly)
            # This is a direct copy of the raw data with explicit orientation for ASTRA
            temp_sino = np.zeros((det_count, angle_count), dtype=np.float32)
            for i in range(angle_count):
                for j in range(det_count):
                    temp_sino[j, i] = sinogram[i, j]
            
            # Create ASTRA data objects
            sino_id = astra.data2d.create('-sino', proj_geom, temp_sino)
            vol_id = astra.data2d.create('-vol', vol_geom)
            
            # Create and configure the FBP reconstruction
            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            cfg['FilterType'] = filter_name
            
            # Run the algorithm
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            
            # Get the result
            reconstruction = astra.data2d.get(vol_id)
            volume[:, :, slice_idx] = reconstruction
            
            # Clean up ASTRA memory
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)
        
        return volume
        
    except Exception as e:
        print(f"ASTRA FBP failed with error: {e}")
        raise e

def sirt_reconstruction_astra(projections, angles, volume_shape, iterations=100):
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
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)
    
    # Initialize volume
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # Handle angle count mismatches gracefully
    if projections.shape[0] != len(angles):
        print(f"WARNING: Projection count ({projections.shape[0]}) doesn't match angles ({len(angles)})")
        # Fix the mismatch by using the smaller count
        if projections.shape[0] > len(angles):
            print(f"Trimming projections from {projections.shape[0]} to {len(angles)}")
            projections = projections[:len(angles)]
        else:
            print(f"Trimming angles from {len(angles)} to {projections.shape[0]}")
            angles = angles[:projections.shape[0]]
            angles_rad = np.deg2rad(angles)
    
    # Process each slice
    try:
        for slice_idx in range(projections.shape[1]):
            # Extract sinogram for current slice
            sinogram = projections[:, slice_idx, :]
            
            # Create ASTRA volume geometry
            vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1])
            
            # Create custom projection vectors
            det_count = sinogram.shape[1]
            angle_count = sinogram.shape[0]
            
            # Create custom projection vectors
            vectors = np.zeros((angle_count, 6))
            for i, angle in enumerate(angles_rad):
                # Ray direction
                vectors[i, 0] = np.cos(angle) 
                vectors[i, 1] = np.sin(angle)
                # Center of detector
                vectors[i, 2] = 0
                vectors[i, 3] = 0  
                # Vector from detector pixel 0 to 1
                vectors[i, 4] = -np.sin(angle)
                vectors[i, 5] = np.cos(angle)
                
            # Create vector-based projection geometry
            proj_geom = astra.create_proj_geom('parallel_vec', det_count, vectors)
            
            # Create a temporary memory block for the sinogram (rotated correctly)
            temp_sino = np.zeros((det_count, angle_count), dtype=np.float32)
            for i in range(angle_count):
                for j in range(det_count):
                    temp_sino[j, i] = sinogram[i, j]
            
            # Create ASTRA data objects
            sino_id = astra.data2d.create('-sino', proj_geom, temp_sino)
            vol_id = astra.data2d.create('-vol', vol_geom)
            
            # Create and configure the SIRT reconstruction
            cfg = astra.astra_dict('SIRT_CUDA')
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            
            # Run the algorithm
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, iterations)
            
            # Get the result
            reconstruction = astra.data2d.get(vol_id)
            volume[:, :, slice_idx] = reconstruction
            
            # Clean up ASTRA memory
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)
        
        return volume
        
    except Exception as e:
        print(f"ASTRA SIRT failed with error: {e}")
        raise e

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
    source_origin_dist = float(geometry['source_origin_dist'])
    origin_detector_dist = float(geometry['origin_detector_dist'])
    detector_width = projections.shape[2]
    detector_height = projections.shape[1]
    
    # Handle angle count mismatches gracefully
    if projections.shape[0] != len(angles_deg):
        print(f"WARNING: Projection count ({projections.shape[0]}) doesn't match angles ({len(angles_deg)})")
        # Fix the mismatch by using the smaller count
        if projections.shape[0] > len(angles_deg):
            print(f"Trimming projections from {projections.shape[0]} to {len(angles_deg)}")
            projections = projections[:len(angles_deg)]
        else:
            print(f"Trimming angles from {len(angles_deg)} to {projections.shape[0]}")
            angles_deg = angles_deg[:projections.shape[0]]
    
    # Convert to radians
    angles_rad = np.deg2rad(angles_deg)
    
    try:
        # Create 3D volume geometry
        vol_geom = astra.create_vol_geom(volume_shape[0], volume_shape[1], volume_shape[2])
        
        # Transpose projections for ASTRA's preferred layout
        projs_astra = np.transpose(projections, (1, 0, 2))
        
        # Create cone-beam vectors
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
        
        # Create cone-beam geometry
        proj_geom = astra.create_proj_geom('cone_vec', detector_height, detector_width, vectors)
        
        # Create projection data
        proj_id = astra.data3d.create('-proj3d', proj_geom, projs_astra)
        
        # Create volume data
        vol_id = astra.data3d.create('-vol', vol_geom)
        
        # Set up the FDK reconstruction
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectionDataId'] = proj_id
        
        # Run the algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # Get the reconstruction
        volume = astra.data3d.get(vol_id)
        
        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)
        
        return volume
        
    except Exception as e:
        print(f"ASTRA FDK failed with error: {e}")
        raise e

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
