# src/visualization.py
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt

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

