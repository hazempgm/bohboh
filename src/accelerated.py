"""
Integration module for tomographic reconstruction with automatic GPU acceleration.
This module selects the fastest available implementation based on hardware.
"""
import os
import platform
import numpy as np

# Check if GPU acceleration is available
GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled using CuPy.")
except ImportError:
    print("CuPy not available. Using CPU implementation.")

# Check if ASTRA Toolbox is available
ASTRA_AVAILABLE = False
try:
    import astra
    ASTRA_AVAILABLE = True
    print("ASTRA Toolbox is available. High-performance GPU reconstruction enabled.")
except ImportError:
    print("ASTRA Toolbox not available. Install with 'conda install -c astra-toolbox astra-toolbox'")

# Import implementations
from reconstruction import (
    filtered_backprojection as filtered_backprojection_cpu,
    art_reconstruction as art_reconstruction_cpu,
    sirt_reconstruction as sirt_reconstruction_cpu,
    fdk_reconstruction as fdk_reconstruction_cpu
)

# Import GPU implementations if available
if GPU_AVAILABLE:
    try:
        from reconstruction_gpu import (
            filtered_backprojection_gpu,
            sirt_reconstruction_gpu,
            fdk_reconstruction_gpu
        )
        print("GPU-accelerated reconstruction algorithms loaded successfully.")
    except ImportError as e:
        print(f"Error loading GPU implementations: {e}")
        GPU_AVAILABLE = False

# Import ASTRA implementations if available
if ASTRA_AVAILABLE:
    try:
        from reconstruction_astra import (
            filtered_backprojection_astra,
            sirt_reconstruction_astra,
            fdk_reconstruction_astra
        )
        print("ASTRA Toolbox reconstruction algorithms loaded successfully.")
    except ImportError as e:
        print(f"Error loading ASTRA implementations: {e}")
        ASTRA_AVAILABLE = False

def filtered_backprojection(projections, angles, volume_shape=None, filter_name='ramp', 
                           use_gpu=None, use_astra=None):
    """
    Filtered backprojection algorithm with automatic acceleration selection.
    
    Args:
        projections (np.ndarray): Preprocessed projection data (angles, height, width).
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple, optional): Shape of the output volume.
        filter_name (str): Filter to use ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann').
        use_gpu (bool, optional): Force using GPU (True) or CPU (False). If None, automatically select.
        use_astra (bool, optional): Force using ASTRA Toolbox. If None, automatically select based on availability.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    # Set default volume shape if not provided
    if volume_shape is None:
        size = projections.shape[2]  # Use width as volume dimension
        volume_shape = (size, size, projections.shape[1])
    
    # Estimate memory requirement
    required_memory_gb = estimate_required_gpu_memory(volume_shape, projections.shape)
    
    # Check if we need to downsample to fit in memory
    original_projections = projections
    original_volume_shape = volume_shape
    downsample_factor = 1
    
    # Get available GPU memory if possible
    available_memory_gb = None
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            available_memory_gb = cp.cuda.runtime.memGetInfo()[0] / (1024**3)
            print(f"Available GPU memory: {available_memory_gb:.2f} GB")
            print(f"Estimated required memory: {required_memory_gb:.2f} GB")
            
            # If required memory exceeds 80% of available memory, downsample
            if required_memory_gb > 0.8 * available_memory_gb:
                # Calculate downsample factor (power of 2 for best results)
                memory_ratio = required_memory_gb / (0.7 * available_memory_gb)
                downsample_factor = 2 ** int(np.ceil(np.log2(np.sqrt(memory_ratio))))
                print(f"Memory requirement too high. Downsampling by factor {downsample_factor}")
                
                # Apply downsampling
                if downsample_factor > 1:
                    from skimage.transform import downscale_local_mean
                    
                    # Downsample projections
                    proj_shape = projections.shape
                    downsampled_projs = np.zeros((
                        proj_shape[0],
                        proj_shape[1] // downsample_factor,
                        proj_shape[2] // downsample_factor
                    ))
                    
                    for i in range(proj_shape[0]):
                        downsampled_projs[i] = downscale_local_mean(
                            projections[i],
                            (downsample_factor, downsample_factor)
                        )
                    
                    projections = downsampled_projs
                    
                    # Update volume shape
                    volume_shape = (
                        original_volume_shape[0] // downsample_factor,
                        original_volume_shape[1] // downsample_factor,
                        original_volume_shape[2] // downsample_factor
                    )
                    
                    print(f"Downsampled projections from {proj_shape} to {projections.shape}")
                    print(f"Downsampled volume from {original_volume_shape} to {volume_shape}")
        except Exception as e:
            print(f"Warning: Could not determine GPU memory. {e}")
    
    # Determine which implementation to use
    should_use_astra = ASTRA_AVAILABLE if use_astra is None else use_astra
    should_use_gpu = GPU_AVAILABLE if use_gpu is None else use_gpu
    
    # Initialize result variable
    result = None
    
    # Prioritize ASTRA if available (it's usually the fastest)
    if should_use_astra and ASTRA_AVAILABLE:
        try:
            print("Using ASTRA Toolbox for Filtered Back Projection (fastest)")
            # Convert filter name for ASTRA if needed
            astra_filter = filter_name
            if filter_name == 'ramp':
                astra_filter = 'ram-lak'
            result = filtered_backprojection_astra(projections, angles, volume_shape, astra_filter)
        except Exception as e:
            print(f"ASTRA acceleration failed: {e}")
            print("Falling back to GPU/CPU implementation")
            should_use_astra = False
    
    # Try GPU implementation next
    if result is None and should_use_gpu and GPU_AVAILABLE and not should_use_astra:
        try:
            print("Using CuPy GPU-accelerated Filtered Back Projection")
            result = filtered_backprojection_gpu(projections, angles, volume_shape, filter_name)
        except Exception as e:
            print(f"GPU acceleration failed: {e}")
            print("Falling back to CPU implementation")
            should_use_gpu = False
    
    # Fall back to CPU implementation
    if result is None and not should_use_gpu and not should_use_astra:
        print("Using CPU-based Filtered Back Projection")
        # For large datasets, we can use chunked processing even on CPU
        if np.prod(projections.shape) > 1e9:  # More than ~1GB
            print("Large dataset detected, using chunked processing")
            from tqdm import tqdm
            
            # Initialize empty volume
            result = np.zeros(volume_shape, dtype=np.float32)
            
            # Process in chunks along the detector rows
            chunk_size = 20  # Process 20 slices at a time
            for start_idx in tqdm(range(0, projections.shape[1], chunk_size)):
                end_idx = min(start_idx + chunk_size, projections.shape[1])
                chunk_projs = projections[:, start_idx:end_idx, :]
                
                # Process the chunk
                chunk_volume = filtered_backprojection_cpu(chunk_projs, angles, 
                                             (volume_shape[0], volume_shape[1], end_idx-start_idx),
                                             filter_name)
                
                # Insert into final volume
                result[:, :, start_idx:end_idx] = chunk_volume
        else:
            result = filtered_backprojection_cpu(projections, angles, volume_shape, filter_name)
    
    # Upsample result if we downsampled earlier
    if downsample_factor > 1 and result is not None:
        print(f"Upsampling result back to original size {original_volume_shape}")
        from scipy.ndimage import zoom
        
        # Calculate zoom factor
        zoom_factor = (
            original_volume_shape[0] / result.shape[0],
            original_volume_shape[1] / result.shape[1],
            original_volume_shape[2] / result.shape[2]
        )
        
        # Upsample
        result = zoom(result, zoom_factor, order=1)
    
    return result

def art_reconstruction(projections, angles, volume_shape, iterations=10, relaxation=0.1, 
                      use_gpu=None, use_astra=None):
    """
    ART reconstruction with automatic acceleration selection.
    
    Args:
        projections (np.ndarray): Preprocessed projection data.
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple): Shape of the output volume.
        iterations (int): Number of iterations.
        relaxation (float): Relaxation parameter.
        use_gpu (bool, optional): Force using GPU (True) or CPU (False). If None, automatically select.
        use_astra (bool, optional): Force using ASTRA Toolbox. If None, automatically select based on availability.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    # ART doesn't have a GPU implementation yet, but ASTRA has equivalent SART
    should_use_astra = ASTRA_AVAILABLE if use_astra is None else use_astra
    
    if should_use_astra and ASTRA_AVAILABLE:
        try:
            print("Using ASTRA Toolbox for ART/SART reconstruction (fastest)")
            # ASTRA doesn't have direct ART, but SIRT is similar for our purposes
            return sirt_reconstruction_astra(projections, angles, volume_shape, iterations)
        except Exception as e:
            print(f"ASTRA acceleration failed: {e}")
            print("Falling back to CPU implementation")
            should_use_astra = False
    
    # Fall back to CPU implementation
    print("Using CPU-based ART reconstruction")
    return art_reconstruction_cpu(projections, angles, volume_shape, iterations, relaxation)

def sirt_reconstruction(projections, angles, volume_shape, iterations=10, 
                       use_gpu=None, use_astra=None):
    """
    SIRT reconstruction with automatic acceleration selection.
    
    Args:
        projections (np.ndarray): Preprocessed projection data.
        angles (np.ndarray): Projection angles in degrees.
        volume_shape (tuple): Shape of the output volume.
        iterations (int): Number of iterations.
        use_gpu (bool, optional): Force using GPU (True) or CPU (False). If None, automatically select.
        use_astra (bool, optional): Force using ASTRA Toolbox. If None, automatically select based on availability.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    # Determine which implementation to use
    should_use_astra = ASTRA_AVAILABLE if use_astra is None else use_astra
    should_use_gpu = GPU_AVAILABLE if use_gpu is None else use_gpu
    
    # Prioritize ASTRA if available
    if should_use_astra and ASTRA_AVAILABLE:
        try:
            print("Using ASTRA Toolbox for SIRT reconstruction (fastest)")
            return sirt_reconstruction_astra(projections, angles, volume_shape, iterations)
        except Exception as e:
            print(f"ASTRA acceleration failed: {e}")
            print("Falling back to GPU/CPU implementation")
            should_use_astra = False
    
    # Try GPU implementation next
    if should_use_gpu and GPU_AVAILABLE and not should_use_astra:
        try:
            print("Using GPU-accelerated SIRT reconstruction")
            return sirt_reconstruction_gpu(projections, angles, volume_shape, iterations)
        except Exception as e:
            print(f"GPU acceleration failed: {e}")
            print("Falling back to CPU implementation")
            should_use_gpu = False
    
    # Fall back to CPU implementation
    if not should_use_gpu and not should_use_astra:
        print("Using CPU-based SIRT reconstruction")
        return sirt_reconstruction_cpu(projections, angles, volume_shape, iterations)

def fdk_reconstruction(projections, geometry, volume_shape, 
                      use_gpu=None, use_astra=None):
    """
    FDK reconstruction with automatic acceleration selection.
    
    Args:
        projections (np.ndarray): Preprocessed projection data.
        geometry (dict): Projection geometry.
        volume_shape (tuple): Shape of the output volume.
        use_gpu (bool, optional): Force using GPU (True) or CPU (False). If None, automatically select.
        use_astra (bool, optional): Force using ASTRA Toolbox. If None, automatically select based on availability.
        
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    # Determine which implementation to use
    should_use_astra = ASTRA_AVAILABLE if use_astra is None else use_astra
    should_use_gpu = GPU_AVAILABLE if use_gpu is None else use_gpu
    
    # Prioritize ASTRA if available
    if should_use_astra and ASTRA_AVAILABLE:
        try:
            print("Using ASTRA Toolbox for FDK reconstruction (fastest)")
            return fdk_reconstruction_astra(projections, geometry, volume_shape)
        except Exception as e:
            print(f"ASTRA acceleration failed: {e}")
            print("Falling back to GPU/CPU implementation")
            should_use_astra = False
    
    # Try GPU implementation next
    if should_use_gpu and GPU_AVAILABLE and not should_use_astra:
        try:
            print("Using GPU-accelerated FDK reconstruction")
            return fdk_reconstruction_gpu(projections, geometry, volume_shape)
        except Exception as e:
            print(f"GPU acceleration failed: {e}")
            print("Falling back to CPU implementation")
            should_use_gpu = False
    
    # Fall back to CPU implementation
    if not should_use_gpu and not should_use_astra:
        print("Using CPU-based FDK reconstruction")
        return fdk_reconstruction_cpu(projections, geometry, volume_shape)

def estimate_required_gpu_memory(volume_shape, projections_shape):
    """
    Estimate the amount of GPU memory required for reconstruction.
    
    Args:
        volume_shape (tuple): Shape of the output volume.
        projections_shape (tuple): Shape of the projection data.
        
    Returns:
        float: Estimated memory requirement in GB.
    """
    # Calculate sizes
    volume_size = np.prod(volume_shape) * 4  # float32 = 4 bytes
    projections_size = np.prod(projections_shape) * 4  # float32 = 4 bytes
    
    # Add overhead for intermediate calculations
    total_bytes = (volume_size + projections_size) * 3
    
    # Convert to GB
    total_gb = total_bytes / (1024**3)
    
    return total_gb

def print_gpu_info():
    """Print information about available GPU resources."""
    print("\n=== GPU Acceleration Information ===")
    
    if not GPU_AVAILABLE and not ASTRA_AVAILABLE:
        print("No GPU acceleration available.")
        return
    
    if GPU_AVAILABLE:
        try:
            # Get device information
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"Found {device_count} CUDA-compatible GPU(s)")
            
            for i in range(device_count):
                cp.cuda.runtime.setDevice(i)
                props = cp.cuda.runtime.getDeviceProperties(i)
                mem_total = cp.cuda.runtime.memGetInfo()[1]
                mem_free = cp.cuda.runtime.memGetInfo()[0]
                
                print(f"\nGPU {i}: {props['name'].decode()}")
                print(f"  CUDA Capability: {props['major']}.{props['minor']}")
                print(f"  Total Memory: {mem_total / 1024**3:.2f} GB")
                print(f"  Free Memory: {mem_free / 1024**3:.2f} GB")
                print(f"  Multi-Processors: {props['multiProcessorCount']}")
        except Exception as e:
            print(f"Error getting GPU information: {e}")
    
    if ASTRA_AVAILABLE:
        print("\nASTRA Toolbox is available for high-performance reconstruction")
        try:
            # Get ASTRA info
            if hasattr(astra, 'get_gpu_info'):
                gpu_info = astra.get_gpu_info()
                print(f"ASTRA GPU Info: {gpu_info}")
        except Exception as e:
            print(f"Error getting ASTRA information: {e}")
    
    print("\nRecommended acceleration method:")
    if ASTRA_AVAILABLE:
        print("  ASTRA Toolbox (fastest, specialized for tomography)")
    elif GPU_AVAILABLE:
        print("  CuPy GPU acceleration (general purpose)")
    else:
        print("  CPU only (slowest)")
    
    print("====================================\n")

# Print acceleration information on module import
print_gpu_info()
