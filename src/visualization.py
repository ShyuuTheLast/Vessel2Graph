# src/visualization.py
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import cv2
import os
import glob

def visualize_skeleton(skeletons, scale_factor=10):
    """
    Visualize skeletons using Mayavi with different colors for each skeleton.

    Parameters:
    - skeletons (dict): Dictionary of skeleton objects.
    - scale_factor (int): Scale factor for the points (default is 10).
    """
    mlab.figure(bgcolor=(1, 1, 1))
    
    # Define a colormap
    colormap = plt.cm.get_cmap('viridis', len(skeletons))
    colors = colormap(np.linspace(0, 1, len(skeletons)))

    # Plot the vertices of the skeletons
    for idx, skeleton in enumerate(skeletons.values()):
        vertices = skeleton.vertices
        color = colors[idx][:3]  # Extract RGB values
        mlab.points3d(vertices[:, 2], vertices[:, 1], vertices[:, 0], mode='sphere', scale_factor=scale_factor, color=tuple(color))
    
    # Show the plot
    mlab.show()

def visualize_radii(skeletons):
    """
    Visualize the distribution of radii in each skeleton.

    Parameters:
    - skeletons (dict): Dictionary of skeleton objects.
    """
    num_skeletons = len(skeletons)
    fig, axes = plt.subplots(num_skeletons, 1, figsize=(10, 6 * num_skeletons))

    if num_skeletons == 1:
        axes = [axes]

    for ax, (label, skeleton) in zip(axes, skeletons.items()):
        radii = skeleton.radius
        ax.hist(radii, bins=50, edgecolor='black')
        ax.set_title(f'Distribution of Radii in Skeleton Nodes (Label: {label})')
        ax.set_xlabel('Radius')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_skeleton_colored(G, attribute, scale_factor=10):
    """
    Visualize the skeleton colored by a specific node attribute using Mayavi.

    Parameters:
    - G (networkx.Graph): The graph containing the skeleton data.
    - attribute (str): The node attribute to color by.
    """
    # Extract node positions and labels, filter nodes without the specified attribute
    node_positions = []
    node_attributes = []
    for node in G.nodes:
        if attribute in G.nodes[node]:
            node_positions.append(node)
            node_attributes.append(G.nodes[node][attribute])

    node_positions = np.array(node_positions)

    # Map attributes to colors
    unique_attributes = list(set(node_attributes))
    color_map = {attr: np.random.rand(3) for attr in unique_attributes}

    # Plot nodes by attribute
    mlab.figure(bgcolor=(1, 1, 1))

    for attr in unique_attributes:
        attr_mask = np.array(node_attributes) == attr
        attr_positions = node_positions[attr_mask]
        color = color_map[attr]
        mlab.points3d(attr_positions[:, 2], attr_positions[:, 1], attr_positions[:, 0],
                      color=tuple(color), mode='sphere', scale_factor=scale_factor)

    mlab.show()

def plot_all_paths_radii(unique_paths):
    """
    Plot histograms for the radii of each path in unique_paths.

    Parameters:
    - unique_paths (list of lists): Each element is a list representing a path, where each node is a tuple (node, radius).
    """
    # Warn the user about potential performance issues with large datasets
    if len(unique_paths) > 50:
        print("Warning: Plotting histograms for a large number of paths may be very slow and memory-intensive.")

    # Plot histogram for the radii of each path
    for idx, path in enumerate(unique_paths):
        radii = [node[1] for node in path]
        plt.figure()
        plt.hist(radii, bins=20, edgecolor='black')
        plt.title(f'Histogram of Radii for Path {idx}')
        plt.xlabel('Radius')
        plt.ylabel('Frequency')
        plt.show()

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