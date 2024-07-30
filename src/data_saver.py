# src/data_saver.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

def save_skeleton_to_file(skeletons, output_path):
    """
    Save the skeletons to a .npz file.

    Parameters:
    - skeletons (dict): A dictionary of skeleton objects, where each key is an ID and each value is a skeleton.
    - output_path (str): The file path to save the .npz file.

    """
    # Check if the output path ends with .npz
    if not output_path.endswith('.npz'):
        if '.' in output_path:
            # If the file has an extension but it's not .npz, prompt the user
            print(f"Warning: The file extension should be .npz. You provided: {output_path}")
            user_input = input("Do you want to change the file extension to .npz? (y/n): ").strip().lower()
            if user_input == 'y':
                output_path = output_path.rsplit('.', 1)[0] + '.npz'
            else:
                print("Proceeding without saving the skeleton.")
                return
        else:
            # No extension provided, add .npz
            output_path += '.npz'
    
    # Save the skeletons to the specified file
    np.savez_compressed(output_path, skeletons=skeletons)
    print(f"Skeletons saved to {output_path}")

def save_segmented_volume(filtered_array, skel_label, file_name):
    """
    Save the segmented volume as an HDF5 file with compression.

    Parameters:
    - filtered_array (np.ndarray): The segmented volume to be saved.
    - skel_label (list): The unique labels in the segmented volume.
    - file_name (str): The name of the output file.

    Returns:
    - None

    Raises:
    - ValueError: If the file name has an extension other than .h5.
    """
    # Ensure the file name ends with .h5
    if '.' in file_name and not file_name.endswith('.h5'):
        raise ValueError("Error: The volume must be saved as an HDF5 file with a .h5 extension.")

    if not file_name.endswith('.h5'):
        file_name += '.h5'

    # Determine the appropriate datatype (e.g., uint8 if labels fit within 0-255)
    max_label = np.max(skel_label)
    if max_label < 256:
        dtype = np.uint8
    elif max_label < 65536:
        dtype = np.uint16
    else:
        dtype = np.uint32

    # Save the full `filtered_array` as a single HDF5 file with compression
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('main', data=filtered_array.astype(dtype), compression='gzip')

    print(f"Full volume saved as: {file_name}")

def save_stats(output_path, branch_points, end_points, neighbor_counts, branch_info, total_length):
    """
    Save statistics to a .npz file.

    Parameters:
    - output_path (str): The file path to save the statistics.
    - branch_points (dict): Dictionary of branch points with coordinates as keys.
    - end_points (dict): Dictionary of end points with coordinates as keys.
    - neighbor_counts (dict): Dictionary with branch point coordinates as keys and neighbor counts as values.
    - branch_info (list): List of dictionaries, each representing a branch with its details.
    - total_length (float): The total length of the skeleton.
    """
    stats = {
        'branch_points': branch_points,
        'end_points': end_points,
        'neighbor_counts': neighbor_counts,
        'branch_info': branch_info,
        'total_length': total_length
    }
    
    # Save the stats dictionary into an .npz file
    np.savez_compressed(output_path, **stats)
    print(f"Statistics saved to {output_path}")

def plot_graph(angle, graph):
    """
    Plot the graph from a specific angle.

    Parameters:
    - angle (float): The angle to view the graph from.
    - graph (networkx.Graph): The graph to plot.

    Returns:
    - matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Extract node positions from the graph
    node_xyz = np.array([node for node in graph.nodes])
    edge_xyz = np.array([(np.array(u), np.array(v)) for u, v in graph.edges()])

    # Plot the nodes
    ax.scatter(node_xyz[:, 2], node_xyz[:, 1], node_xyz[:, 0], s=100, ec="w", label="nodes")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(vizedge[:, 2], vizedge[:, 1], vizedge[:, 0], color="tab:gray")

    # Set the viewing angle
    ax.view_init(elev=20, azim=angle)

    def set_ticks(ax):
        """Set ticks on each axis evenly spread across the range."""
        for dim, values in zip((ax.xaxis, ax.yaxis, ax.zaxis), (node_xyz[:, 2], node_xyz[:, 1], node_xyz[:, 0])):
            min_val, max_val = values.min(), values.max()
            ticks = np.linspace(min_val, max_val, num=10)
            dim.set_ticks(ticks)
            dim.set_tick_params(labelsize=10)

    set_ticks(ax)

    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    return fig

def graph2video(graph, output_filename, num_rotations=3, fps=10):
    """
    Create a video from a graph by plotting it from multiple angles.

    Parameters:
    - graph (networkx.Graph): The graph to visualize.
    - output_filename (str): Path to save the output video.
    - num_rotations (int): Number of full rotations for the video (default is 3).
    - fps (int): Frames per second for the video (default is 10).

    Raises:
    - ValueError: If the file name has an extension other than .avi or .mp4.
    """
    # Ensure the file name ends with .avi or .mp4
    if not (output_filename.endswith('.avi') or output_filename.endswith('.mp4')):
        raise ValueError("Error: The video must be saved as an AVI or MP4 file with a .avi or .mp4 extension.")

    # Determine the codec based on the file extension
    if output_filename.endswith('.avi'):
        codec = cv2.VideoWriter_fourcc(*'DIVX')
    elif output_filename.endswith('.mp4'):
        codec = cv2.VideoWriter_fourcc(*'mp4v')

    # Calculate the number of frames and angles
    num_frames = num_rotations * 60
    angles = np.linspace(0, 360 * num_rotations, num_frames)

    # Generate plots from multiple angles and save them as images
    for angle in angles:
        fig = plot_graph(angle, graph)
        filename = f"temp_{angle:.0f}.png"
        fig.savefig(filename)
        plt.close(fig)  # Close the figure to save memory

    # Path to the temporary PNG files
    png_files = sorted(glob.glob("temp_*.png"), key=os.path.getmtime)

    # Read the first image to get the size
    frame = cv2.imread(png_files[0])
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    video = cv2.VideoWriter(output_filename, codec, fps, (width, height))

    for filename in png_files:
        img = cv2.imread(filename)
        video.write(img)

    # Release the video writer
    video.release()

    # Remove the temporary files
    for filename in png_files:
        os.remove(filename)
        
