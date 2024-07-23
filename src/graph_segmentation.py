# src/graph_segmentation.py
import os
import kimimaro
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.ndimage as ndi
import h5py

def skeletonize_volume(volume, teasar_params, anisotropy):
    """
    Skeletonize the given volume using Kimimaro.

    Parameters:
    - volume (np.ndarray): The volume to be skeletonized.
    - teasar_params (dict): The parameters for the TEASAR algorithm.
    - anisotropy (tuple): The anisotropy scaling factors.
    - parallel (int): The number of parallel processes to use (default is 1).

    Returns:
    - dict: The skeletonized volume.
    """
    skeletons = kimimaro.skeletonize(
        volume,
        teasar_params=teasar_params,
        anisotropy=anisotropy,
        parallel=1
    )
    return skeletons

def full_graph_generation(skeletons):
    """
    Generate a full graph from the skeletons.

    Parameters:
    - skeletons (dict): Dictionary of skeleton objects.

    Returns:
    - networkx.Graph: The generated graph.
    """
    G = nx.Graph()

    for seg_id, skeleton in skeletons.items():
        vertices = {idx: tuple(vertex) for idx, vertex in enumerate(skeleton.vertices)}
        radii = skeleton.radius
        for vertex_id, vertex in vertices.items():
            G.add_node(vertex, seg_id=seg_id, radius=radii[vertex_id])  # Track which node belongs to which segmentation ID
        for edge in skeleton.edges:
            G.add_edge(vertices[edge[0]], vertices[edge[1]])

    return G

def get_branch_points(G):
    """
    Identify branch points in the graph with radius attribute.

    Parameters:
    - G (networkx.Graph): The graph to analyze.

    Returns:
    - dict: Dictionary of branch points with nodes as keys and (seg_id, radius) as values.
    """
    branch_points = {node: (G.nodes[node]['seg_id'], G.nodes[node]['radius']) for node in G.nodes if len(list(G.neighbors(node))) > 2}
    return branch_points

def get_end_points(G):
    """
    Identify end points in the graph with radius attribute.

    Parameters:
    - G (networkx.Graph): The graph to analyze.

    Returns:
    - dict: Dictionary of end points with nodes as keys and (seg_id, radius) as values.
    """
    end_points = {node: (G.nodes[node]['seg_id'], G.nodes[node]['radius']) for node in G.nodes if len(list(G.neighbors(node))) == 1}
    return end_points

def traverse_path(G, start, previous, branch_points, end_points):
    """
    Trace a path from a starting point to another branch point or an end point.

    Parameters:
    - G (networkx.Graph): The full graph.
    - start (tuple): The starting node.
    - previous (tuple): The previous node in the path.
    - branch_points (dict): Dictionary of branch points.
    - end_points (dict): Dictionary of end points.

    Returns:
    - tuple: The path and the endpoint.
    """
    current = start
    path = [[current, G.nodes[current]['radius']]]

    while True:
        neighbors = list(G.neighbors(current))
        next_node = neighbors[0] if neighbors[1] == previous else neighbors[1]

        if next_node in branch_points or next_node in end_points:
            return path, next_node

        previous = current
        current = next_node
        path.append([current, G.nodes[current]['radius']])


def simplified_graph_generation(G, branch_points, end_points):
    """
    Generate a simplified graph containing only branch points and end points from the full graph.

    Parameters:
    - G (networkx.Graph): The full graph.
    - branch_points (dict): Dictionary of branch points.
    - end_points (dict): Dictionary of end points.

    Returns:
    - tuple: The simplified graph and the list of unique paths.
    """
    simplified_G = nx.Graph()

    # Add branch points and end points to the simplified graph with seg_id
    for node, (seg_id, radius) in branch_points.items():
        simplified_G.add_node(node, seg_id=seg_id, radius=radius)

    for node, (seg_id, radius) in end_points.items():
        simplified_G.add_node(node, seg_id=seg_id, radius=radius)

    paths = []

    # Trace paths starting from each branch point
    for branch in branch_points.keys():
        neighbors = list(G.neighbors(branch))
        for neighbor in neighbors:
            if neighbor not in branch_points and neighbor not in end_points:
                path, endpoint = traverse_path(G, neighbor, branch, branch_points, end_points)
                simplified_G.add_edge(branch, endpoint)
                paths.append(path)

    # Remove duplicate paths
    unique_paths = []
    for path in paths:
        path_length = len(path)
        start, end = path[0], path[-1]
        if not any(len(p) == path_length and (p[0] == start and p[-1] == end or p[0] == end and p[-1] == start) for p in unique_paths):
            unique_paths.append(path)

    return simplified_G, unique_paths

def get_median_radii(unique_paths):
    """
    Get the median radius for each branch in unique_paths.

    Parameters:
    - unique_paths (list of lists): Each element is a list representing a path, where each node is a tuple (node, radius).

    Returns:
    - list: List of median radii for each branch.
    """
    medians = []
    for path in unique_paths:
        radii = [node[1] for node in path]
        median_radius = np.median(radii)
        medians.append(median_radius)
    return medians

def plot_elbow_curve(medians, max_clusters=10):
    """
    Plot the Elbow curve to help determine the optimal number of clusters.

    Parameters:
    - medians (list): List of median radii for each branch.
    - max_clusters (int): Maximum number of clusters to consider (default is 10).
    """
    # Set the environment variable to avoid memory leaks
    os.environ["OMP_NUM_THREADS"] = "1"

    inertia = []
    range_n_clusters = list(range(1, max_clusters + 1))
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(np.array(medians).reshape(-1, 1))
        inertia.append(kmeans.inertia_)

    # Plot the Elbow curve
    plt.figure()
    plt.plot(range_n_clusters, inertia, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()
    
    # Ask the user to choose the optimal number of clusters
    print("Please choose the optimal number of clusters based on the Elbow plot.")

def cluster_medians(medians, optimal_clusters):
    """
    Cluster the medians based on the chosen number of clusters.

    Parameters:
    - medians (list): List of median radii for each branch.
    - optimal_clusters (int): The chosen number of clusters.

    Returns:
    - tuple: Cluster labels for each median and the unique labels.
    """
    # Set the environment variable to avoid memory leaks
    os.environ["OMP_NUM_THREADS"] = "1"

    kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42).fit(np.array(medians).reshape(-1, 1))
    labels = kmeans.labels_ + 1  # Add 1 to each label
    return labels

def relabel_graph_with_branches(G, unique_paths, labels):
    """
    Relabel the full graph with branch indices and cluster labels.

    Parameters:
    - G (networkx.Graph): The full graph.
    - unique_paths (list of lists): Each element is a list representing a path, where each node is a tuple (node, radius).
    - labels (list): Cluster labels for each median.

    Returns:
    - networkx.Graph: The relabeled graph.
    """
    for index, unique_path in enumerate(unique_paths):
        label = labels[index]
        for node, radius in unique_path:
            G.nodes[node]['label'] = label
            G.nodes[node]['branch'] = index + 1
    return G

def segment_volume(filtered_array, G, scaled_voxel_size, attribute='label'):
    """
    Segment the volume using distance transforming with a specified node attribute.

    Parameters:
    - filtered_array (np.ndarray): The filtered array to be segmented.
    - G (networkx.Graph): The graph containing the skeleton data with the specified attribute.
    - scaled_voxel_size (tuple): The anisotropy scaling factors.
    - attribute (str): The node attribute to segment by (default is 'label').

    Returns:
    - tuple: The segmented volume and the unique labels.
    """
    # Create a 3D volume for the skeleton points with the same shape as `filtered_array`
    vol_skel = np.zeros_like(filtered_array)
    skel_label = set()

    # Populate `vol_skel` with labels from graph nodes, excluding branch and end points
    for node in G.nodes:
        if attribute in G.nodes[node]:  # Ensure the node has the attribute
            # Adjust coordinates to fit the original array size
            adjusted_node = tuple(int(coord / scale) for coord, scale in zip(node, scaled_voxel_size))
            vol_skel[adjusted_node] = G.nodes[node][attribute]
            skel_label.add(G.nodes[node][attribute])

    skel_label = list(skel_label)

    # Initialize output volumes
    output_label = np.zeros_like(filtered_array)
    output_dist = np.ones_like(filtered_array) * 99999

    # Compute distance transforms and update the output volumes
    for label in skel_label:
        mask = vol_skel == label
        print(f"Label {label} has {np.sum(mask)} voxels")

        dist = ndi.distance_transform_cdt(~mask)  # Use Euclidean distance transform for accuracy
        update = dist < output_dist
        print(f"Updating {np.sum(update)} voxels")

        # Update the output arrays
        output_label[update] = label
        output_dist[update] = dist[update]

    # Apply the labels to the segmented volume
    filtered_array[filtered_array == 1] = output_label[filtered_array == 1]

    return filtered_array, skel_label

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

def scale_graph(G, common_factor):
    """
    Scale the graph to the actual size of the data represented by the isotropy.

    Parameters:
    - G (networkx.Graph): The original graph.
    - common_factor (float): The common factor by which to scale the graph.

    Returns:
    - networkx.Graph: The scaled graph.
    """
    scaled_G = nx.Graph()

    # Scale the node coordinates and radius
    for node, data in G.nodes(data=True):
        scaled_node = tuple(coord * common_factor for coord in node)
        scaled_radius = data['radius'] * common_factor
        scaled_G.add_node(scaled_node, seg_id=data['seg_id'], radius=scaled_radius)
        
    # Add the edges
    for u, v in G.edges:
        scaled_u = tuple(coord * common_factor for coord in u)
        scaled_v = tuple(coord * common_factor for coord in v)
        scaled_G.add_edge(scaled_u, scaled_v)

    return scaled_G
