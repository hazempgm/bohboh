"""
Utility functions for tomographic reconstruction.
"""
import os
import numpy as np
import tifffile
import vtk
from vtk.util import numpy_support
import re
import json
import configparser

def ensure_directory(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Path to directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_tiff_stack(directory, pattern="*.tif*"):
    """
    Load a stack of TIFF images from a directory.
    
    Args:
        directory (str): Directory containing TIFF images.
        pattern (str): Glob pattern for selecting TIFF files.
        
    Returns:
        tuple: (projections, filenames)
            projections: 3D array containing all projections.
            filenames: List of filenames in order they were loaded.
    """
    import glob
    
    # Get sorted list of TIFF files
    tiff_files = sorted(glob.glob(os.path.join(directory, pattern)))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {directory} with pattern {pattern}")
    
    # Load first image to get dimensions
    img = tifffile.imread(tiff_files[0])
    
    # Allocate array for all projections
    projections = np.zeros((len(tiff_files), img.shape[0], img.shape[1]), dtype=img.dtype)
    
    # Load all projections
    for i, tiff_file in enumerate(tiff_files):
        projections[i] = tifffile.imread(tiff_file)
    
    return projections, tiff_files

def read_metadata_file(filepath):
    """
    Read metadata from a text file.
    
    Args:
        filepath (str): Path to metadata file.
        
    Returns:
        dict: Metadata dictionary.
    """
    metadata = {}
    
    # Check if file exists
    if not os.path.exists(filepath):
        return metadata
    
    # Try to determine file format and parse accordingly
    file_ext = os.path.splitext(filepath)[1].lower()
    
    # Handle JSON metadata files
    if file_ext == '.json':
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            # Process the JSON data
            metadata = json_data
            
            # Extract key geometry information for easier access
            if 'geometry' in metadata:
                geom = metadata['geometry']
                
                # Create a simpler geometry structure for direct access
                simple_geom = {}
                
                # Get distances
                if 'distanceSourceObject' in geom:
                    simple_geom['source_origin_dist'] = float(geom['distanceSourceObject'])
                
                if 'distanceObjectDetector' in geom:
                    simple_geom['origin_detector_dist'] = float(geom['distanceObjectDetector'])
                elif 'distanceSourceDetector' in geom and 'distanceSourceObject' in geom:
                    source_detector = float(geom['distanceSourceDetector'])
                    source_origin = float(geom['distanceSourceObject'])
                    simple_geom['origin_detector_dist'] = source_detector - source_origin
                
                # Get detector information
                if 'detectorPixel' in geom:
                    simple_geom['detector_shape'] = tuple(geom['detectorPixel'])
                
                # Store in metadata
                metadata['geometry_simplified'] = simple_geom
            
            return metadata
        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {filepath}: {e}")
            return metadata
    
    # Handle INI-style config files
    with open(filepath, 'r') as f:
        content = f.read().strip()
        
        # Check if it's an INI-style file with sections like [GENERAL], [GEOMETRY]
        if re.search(r'^\[(.*?)\]', content, re.MULTILINE):
            try:
                config = configparser.ConfigParser()
                config.read(filepath)
                
                # Convert ConfigParser object to dictionary
                for section in config.sections():
                    metadata[section] = {}
                    for key, value in config[section].items():
                        # Try to convert value to numeric if possible
                        try:
                            if '.' in value:
                                metadata[section][key] = float(value)
                            elif value.isdigit():
                                metadata[section][key] = int(value)
                            else:
                                metadata[section][key] = value
                        except ValueError:
                            metadata[section][key] = value
                
                # Extract key geometry information for easier access
                if 'GEOMETRY' in metadata:
                    geom = metadata['GEOMETRY']
                    metadata['geometry'] = {}
                    
                    # Get the distance values
                    if 'distancesourcedetector' in {k.lower(): v for k, v in geom.items()}:
                        source_detector_dist = next(v for k, v in geom.items() 
                                                  if k.lower() == 'distancesourcedetector')
                        metadata['source_detector_distance'] = source_detector_dist
                        
                    if 'distancesourceorigin' in {k.lower(): v for k, v in geom.items()}:
                        source_origin_dist = next(v for k, v in geom.items() 
                                                if k.lower() == 'distancesourceorigin')
                        metadata['source_object_distance'] = source_origin_dist
                        metadata['geometry']['source_origin_dist'] = source_origin_dist
                        
                        # Calculate origin-to-detector distance if source-to-detector is available
                        if 'source_detector_distance' in metadata:
                            origin_detector_dist = metadata['source_detector_distance'] - source_origin_dist
                            metadata['geometry']['origin_detector_dist'] = origin_detector_dist
                    
                # Extract angle information
                if 'ACQUISITION' in metadata:
                    acq = metadata['ACQUISITION']
                    
                    # Attempt to extract angle information
                    angle_first = None
                    angle_last = None
                    angle_interval = None
                    num_images = None
                    
                    # Extract parameters with case-insensitive keys
                    acq_lower = {k.lower(): v for k, v in acq.items()}
                    
                    if 'anglefirst' in acq_lower:
                        angle_first = acq_lower['anglefirst']
                    if 'anglelast' in acq_lower:
                        angle_last = acq_lower['anglelast']
                    if 'angleinterval' in acq_lower:
                        angle_interval = acq_lower['angleinterval']
                    if 'numberimages' in acq_lower:
                        num_images = acq_lower['numberimages']
                    
                    # Store in metadata
                    metadata['angle_info'] = {
                        'first': angle_first,
                        'last': angle_last,
                        'interval': angle_interval,
                        'count': num_images
                    }
                    
                    # Generate angles array if possible
                    if angle_first is not None and num_images is not None:
                        if angle_interval is not None:
                            # Use interval if provided
                            angles = np.arange(angle_first, angle_first + num_images * angle_interval, angle_interval)
                            angles = angles[:num_images]  # Ensure correct length
                        elif angle_last is not None:
                            # Use first and last if provided
                            angles = np.linspace(angle_first, angle_last, num_images)
                        else:
                            # Default to evenly spaced over 0-360 degrees
                            angles = np.linspace(0, 360, num_images, endpoint=False)
                        
                        metadata['angles'] = angles
            except Exception as e:
                print(f"Error parsing INI file {filepath}: {e}")
        
        # Handle simple key-value pair files
        else:
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key, value = parts
                        key = key.strip()
                        value = value.strip()
                        
                        # Try to convert value to numeric if possible
                        try:
                            if '.' in value:
                                metadata[key] = float(value)
                            elif value.isdigit():
                                metadata[key] = int(value)
                            else:
                                metadata[key] = value
                        except ValueError:
                            metadata[key] = value
    
    return metadata

def find_metadata_for_tiff(tiff_filepath):
    """
    Find and read metadata file associated with a TIFF file.
    
    Args:
        tiff_filepath (str): Path to TIFF file.
        
    Returns:
        dict: Metadata dictionary.
    """
    # List of potential metadata file extensions
    meta_extensions = ['.txt', '.meta', '.metadata', '.json', '.ini', '.cfg']
    
    # Base path without extension
    base_path = os.path.splitext(tiff_filepath)[0]
    
    # Try each metadata extension
    for ext in meta_extensions:
        meta_path = base_path + ext
        if os.path.exists(meta_path):
            return read_metadata_file(meta_path)
    
    # If no file-specific metadata found, look in the same directory
    directory = os.path.dirname(tiff_filepath)
    for filename in os.listdir(directory):
        if filename.lower().startswith('metadata') and os.path.splitext(filename)[1] in meta_extensions:
            return read_metadata_file(os.path.join(directory, filename))
    
    # No metadata found
    return {}

def extract_angles_from_json(metadata):
    """
    Extract rotation angles from JSON metadata.
    
    Args:
        metadata (dict): JSON metadata dictionary.
        
    Returns:
        np.ndarray: Array of angles in degrees.
    """
    if not metadata or 'geometry' not in metadata:
        return None
    
    geom = metadata['geometry']
    
    # Extract projection angles if available
    if 'projectionAngles' in geom:
        angle_data = geom['projectionAngles']
        angles = np.array([item['angle'] for item in angle_data])
        
        # Check if angles are in radians (usually if max value is small)
        if np.max(angles) < 7:
            print("Converting angles from radians to degrees")
            angles = np.rad2deg(angles)
        
        return angles
    
    # Try alternative sources
    if 'totalAngle' in geom and 'projections' in metadata and 'numProjections' in metadata['projections']:
        total_angle = geom['totalAngle']
        num_projections = metadata['projections']['numProjections']
        
        # Create evenly spaced angles
        angles = np.linspace(0, total_angle, num_projections, endpoint=False)
        return angles
    
    return None

def extract_angles_from_metadata(tiff_files):
    """
    Extract rotation angles from metadata files associated with TIFF files.
    
    Args:
        tiff_files (list): List of TIFF file paths.
        
    Returns:
        np.ndarray: Array of angles in degrees.
    """
    # First check if there's a common metadata file in the directory
    directory = os.path.dirname(tiff_files[0])
    common_metadata = None
    
    # Check for common metadata files
    for filename in ['metadata.txt', 'metadata.json', 'scan_info.txt']:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            common_metadata = read_metadata_file(filepath)
            break
    
    # If common metadata has angles, use them
    if common_metadata and 'angles' in common_metadata:
        return common_metadata['angles']
    
    # If we have JSON metadata, try to extract angles
    if common_metadata and 'geometry' in common_metadata:
        angles = extract_angles_from_json(common_metadata)
        if angles is not None:
            return angles
    
    # If no common angles found, check individual metadata files
    angles = []
    for tiff_file in tiff_files:
        metadata = find_metadata_for_tiff(tiff_file)
        angle = None
        
        # Try to find angle information
        if 'angle' in metadata:
            angle = metadata['angle']
        elif 'Angle' in metadata:
            angle = metadata['Angle']
        elif 'ANGLE' in metadata:
            angle = metadata['ANGLE']
        elif 'rotation_angle' in metadata:
            angle = metadata['rotation_angle']
        
        angles.append(angle)
    
    return np.array(angles)

def extract_angles_from_filenames(filenames, pattern='_(\d+)deg'):
    """
    Extract rotation angles from filenames using regex pattern.
    
    Args:
        filenames (list): List of filenames.
        pattern (str): Regex pattern to extract angle.
        
    Returns:
        np.ndarray: Array of angles in degrees.
    """
    angles = []
    regex = re.compile(pattern)
    
    for filename in filenames:
        # Get just the filename without the path
        base_filename = os.path.basename(filename)
        match = regex.search(base_filename)
        
        if match:
            try:
                angle = float(match.group(1))
                angles.append(angle)
            except ValueError:
                angles.append(None)
        else:
            angles.append(None)
    
    # If all angles are None, try to create evenly spaced angles
    if all(a is None for a in angles):
        return np.linspace(0, 360, len(filenames), endpoint=False)
    
    return np.array(angles)

def load_projections_with_metadata(directory, pattern="*.tif*", angle_pattern='_(\d+)deg'):
    """
    Load projections and associated metadata.
    
    Args:
        directory (str): Directory containing projection images.
        pattern (str): Glob pattern for selecting TIFF files.
        angle_pattern (str): Regex pattern to extract angle from filename if not in metadata.
        
    Returns:
        tuple: (projections, angles, metadata)
            projections: 3D array containing all projections.
            angles: Array of projection angles in degrees.
            metadata: Dictionary of additional metadata.
    """
    # Load projection images
    projections, filenames = load_tiff_stack(directory, pattern)
    
    # Check for metadata files in directory
    metadata = {}
    for meta_file in ['metadata.txt', 'metadata.json', 'scan_info.txt']:
        metadata_path = os.path.join(directory, meta_file)
        if os.path.exists(metadata_path):
            metadata = read_metadata_file(metadata_path)
            print(f"Found metadata file: {metadata_path}")
            break
    
    # Try to get angles from JSON metadata first if available
    if 'geometry' in metadata:
        angles = extract_angles_from_json(metadata)
        if angles is not None:
            return projections, angles, metadata
    
    # Try to get angles from standard metadata format
    angles = extract_angles_from_metadata(filenames)
    
    # If all angles are None, try to extract from filenames
    if np.all(angles == None):
        angles = extract_angles_from_filenames(filenames, angle_pattern)
    
    # Get geometry information from metadata if available
    if 'geometry' not in metadata and 'source_detector_distance' in metadata and 'source_object_distance' in metadata:
        source_origin_dist = metadata.get('source_object_distance')
        origin_detector_dist = metadata.get('source_detector_distance') - source_origin_dist
        
        metadata['geometry'] = {
            'source_origin_dist': source_origin_dist,
            'origin_detector_dist': origin_detector_dist
        }
    
    return projections, angles, metadata

def create_projection_geometry(angles, detector_shape, source_origin_dist, origin_detector_dist=None):
    """
    Create a dictionary containing the projection geometry.
    
    Args:
        angles (np.ndarray): Array of projection angles in degrees.
        detector_shape (tuple): Shape of detector (height, width).
        source_origin_dist (float): Distance from source to rotation center.
        origin_detector_dist (float, optional): Distance from rotation center to detector.
            If None, equal to source_origin_dist (symmetric).
            
    Returns:
        dict: Projection geometry parameters.
    """
    if origin_detector_dist is None:
        origin_detector_dist = source_origin_dist
        
    return {
        'angles': angles,
        'detector_shape': detector_shape,
        'source_origin_dist': source_origin_dist,
        'origin_detector_dist': origin_detector_dist,
        'total_dist': source_origin_dist + origin_detector_dist
    }

def save_numpy_as_vtk(volume, filename, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Save a 3D NumPy array as a VTK file for 3D visualization.
    
    Args:
        volume (np.ndarray): 3D volume data.
        filename (str): Output filename (.vti extension recommended).
        spacing (tuple): Voxel spacing in (x, y, z).
        origin (tuple): Volume origin coordinates.
    """
    # Ensure the data is in float32 format
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)
    
    # Create VTK image data
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(volume.shape[2], volume.shape[1], volume.shape[0])
    vtk_data.SetSpacing(spacing)
    vtk_data.SetOrigin(origin)
    
    # Convert NumPy array to VTK array
    flat_data = volume.ravel(order='F')
    vtk_array = numpy_support.numpy_to_vtk(flat_data)
    
    # Add array to image data
    vtk_data.GetPointData().SetScalars(vtk_array)
    
    # Write VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_data)
    writer.Write()
    
def save_volume_as_tiff_stack(volume, output_dir, base_filename="slice"):
    """
    Save a 3D volume as a stack of TIFF images.
    
    Args:
        volume (np.ndarray): 3D volume data.
        output_dir (str): Output directory for TIFF stack.
        base_filename (str): Base filename for each slice.
    """
    ensure_directory(output_dir)
    
    # Normalize volume to 0-65535 for 16-bit TIFF
    if volume.dtype != np.uint16:
        v_min, v_max = volume.min(), volume.max()
        if v_min != v_max:  # Avoid division by zero
            volume_norm = ((volume - v_min) / (v_max - v_min) * 65535).astype(np.uint16)
        else:
            volume_norm = np.zeros_like(volume, dtype=np.uint16)
    else:
        volume_norm = volume
    
    # Save each slice as a TIFF file
    for i in range(volume.shape[0]):
        filename = os.path.join(output_dir, f"{base_filename}_{i:04d}.tiff")
        tifffile.imwrite(filename, volume_norm[i])

def calculate_optimal_volume_shape(projections, metadata=None):
    """
    Calculate optimal volume shape based on projection dimensions and metadata.
    
    Args:
        projections (np.ndarray): Projection data with shape (angles, height, width)
        metadata (dict, optional): Metadata dictionary with geometry information
        
    Returns:
        tuple: Optimal volume shape (width, height, depth)
    """
    # Default: use detector width for x/y dimensions, and height for z
    det_width = projections.shape[2]
    det_height = projections.shape[1]
    
    # Check if we have object bounding box info in metadata
    if metadata and 'geometry' in metadata and 'objectBoundingBox' in metadata['geometry']:
        bbox = metadata['geometry']['objectBoundingBox']
        if 'sizeXYZ' in bbox:
            # Get the relative size of the object
            size_xyz = bbox['sizeXYZ']
            if isinstance(size_xyz, list) and len(size_xyz) == 3:
                # Use the bounding box to determine the aspect ratio
                aspect_x = size_xyz[0] / max(size_xyz)
                aspect_y = size_xyz[1] / max(size_xyz)
                aspect_z = size_xyz[2] / max(size_xyz)
                
                # Scale the detector dimensions by the aspect ratio
                # This preserves the object's proportions
                max_dim = max(det_width, det_height)
                width = int(max_dim * aspect_x)
                height = int(max_dim * aspect_y)
                depth = int(max_dim * aspect_z)
                
                return (width, height, depth)
    
    # If no useful object info, use detector dimensions
    # Use detector width as x/y size - this is common in tomography
    # and prevents massive overallocation for asymmetric detectors
    return (det_width, det_width, det_height)

def calculate_memory_requirements(volume_shape, dtype=np.float32):
    """
    Calculate memory requirements for a volume with given shape.
    
    Args:
        volume_shape (tuple): Shape of the volume (width, height, depth)
        dtype (np.dtype): Data type of the volume
        
    Returns:
        float: Memory requirement in GB
    """
    # Get bytes per element for the dtype
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Calculate total memory
    total_elements = np.prod(volume_shape)
    total_bytes = total_elements * bytes_per_element
    
    # Convert to GB
    total_gb = total_bytes / (1024**3)
    
    return total_gb
