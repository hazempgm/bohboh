import pyvista as pv
import os

# Define the path to the saved .vtk file
vtk_volume_path = os.path.join('/home/syscall/Downloads/reconstructed_volume_512.vtk')

if os.path.exists(vtk_volume_path):
    print(f"Reading .vtk file from: {vtk_volume_path}")
    try:
        # Read the VTK file
        vtk_volume_data = pv.read(vtk_volume_path)
        print("File read successfully.")

        # The data is typically stored as a PyVista DataSet or ImageData object
        # You can access the numpy array data like this:
        # numpy_volume = vtk_volume_data.contour().extract_surface().clean().extract_surface().points # Example for point data
        # For ImageData, you might access point_data or cell_data
        # numpy_volume = vtk_volume_data.point_data['values'].reshape(vtk_volume_data.dimensions) # Example for point data

        # To inspect the data structure:
        print("\nVTK data object information:")
        print(vtk_volume_data)

        # If the data is stored as point data in an ImageData object, you might access it like:
        if hasattr(vtk_volume_data, 'point_data') and 'ImageScalars' in vtk_volume_data.point_data:
             numpy_volume_from_vtk = vtk_volume_data.point_data['ImageScalars'].reshape(vtk_volume_data.dimensions, order='F') # Use 'F' order if VTK saves Fortran order
             print(f"\nSuccessfully extracted numpy array with shape: {numpy_volume_from_vtk.shape}")
             # You can now work with numpy_volume_from_vtk as a standard numpy array
             print(f"Min value: {numpy_volume_from_vtk.min()}, Max value: {numpy_volume_from_vtk.max()}")
             # Note: The shape might need to be transposed depending on how vedo saved it
             # If the shape is (Z, Y, X) but you need (X, Y, Z) for plotting:
             # numpy_volume_from_vtk = numpy_volume_from_vtk.transpose(2, 1, 0)


    except Exception as e:
        print(f"Error reading .vtk file: {e}")
else:
    print(f"Error: .vtk file not found at {vtk_volume_path}")


# Attempt to visualize the VTK data object using PyVista
if 'vtk_volume_data' in locals() and vtk_volume_data is not None:
    print("Attempting to plot VTK data using PyVista...")
    try:
        # PyVista's default plot method for ImageData often does a volume rendering
        vtk_volume_data.plot(volume=True)
        print("PyVista plot command executed.")
        # Note: The plot might appear below this cell or in a separate window depending on your environment.
    except Exception as e:
        print(f"Error during PyVista plotting: {e}")
else:
    print("VTK data object 'vtk_volume_data' is not available for plotting.")