# src/graph_segmentation.py
import os
import kimimaro
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.ndimage as ndi

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

def get_neighbor_counts(G, branch_points):
    """
    Generate a dictionary with branch point coordinates as keys and the number of their neighbors as values.

    Parameters:
    - G (networkx.Graph): The graph object containing the nodes and edges.
    - branch_points (dict): Dictionary of branch points.

    Returns:
    - dict: A dictionary with branch point coordinates as keys and neighbor counts as values.
    """
    neighbor_counts = {}
    for point in branch_points.keys():
        neighbor_counts[point] = len(list(G.neighbors(point)))
    return neighbor_counts

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

def relabel_graph_with_branches(G, unique_paths, labels, radii):
    """
    Relabel the graph nodes with branch indices and clusters, and calculate branch details.

    Parameters:
    - G (networkx.Graph): The original graph.
    - unique_paths (list): A list of unique paths in the graph, each containing tuples of (coords, radius).
    - labels (list): The cluster labels for the branches.
    - radii (list): The median radii for each branch.

    Returns:
    - networkx.Graph: The graph with nodes relabeled.
    - list: A list of dictionaries, each representing a branch with its details.
    - float: The total length of the skeleton.
    """
    branch_info = []
    total_length = 0

    for index, path in enumerate(unique_paths):
        branch_details = {
            "index": index + 1,  # 1-based index
            "coords": [],
            "median_radius": radii[index],
            "label": labels[index],
            "length": 0
        }

        previous_node = None
        for node, radius in path:
            G.nodes[node]['label'] = labels[index]
            G.nodes[node]['branch'] = index + 1
            
            branch_details["coords"].append(node)
            
            # If there is a previous node, calculate the length to the current node
            if previous_node is not None:
                branch_length = np.linalg.norm(np.array(previous_node) - np.array(node))
                branch_details["length"] += branch_length
            
            previous_node = node
        
        # Add the full branch length to the total length after the branch is processed
        total_length += branch_details["length"]
        branch_info.append(branch_details)

    return G, branch_info, total_length

def get_ellipsoid_surface(radius, scaling_factors):
    """
    Generate the surface voxels of an ellipsoid given a radius and scaling factors.

    Parameters:
    - radius (int): The radius of the ellipsoid.
    - scaling_factors (tuple): The scaling factors (sz, sy, sx) for the z, y, x axes.

    Returns:
    - np.ndarray: An array of shape (N, 3) with the coordinates of the surface voxels.
    """
    sz, sy, sx = scaling_factors
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    # Parametric equations for the ellipsoid
    x = radius * np.sin(theta) * np.cos(phi) / sx
    y = radius * np.sin(theta) * np.sin(phi) / sy
    z = radius * np.cos(theta) / sz

    # Flatten and combine coordinates
    coords = np.vstack((z.ravel(), y.ravel(), x.ravel())).T
    return coords

def segment_volume(filtered_array, G, scaled_voxel_size, attribute='label'):
    """
    Segment the volume using distance transforming with a specified node attribute.

    Parameters:
    - filtered_array (np.ndarray): The filtered array to be segmented.
    - G (networkx.Graph): The graph containing the skeleton data with the specified attribute.
    - scaled_voxel_size (tuple): The anisotropy scaling factors (sz, sy, sx).
    - attribute (str): The node attribute to segment by (default is 'label').

    Returns:
    - tuple: The segmented volume and the unique labels.
    """
    # Create a 3D volume for the skeleton points with the same shape as `filtered_array`
    category_indices = np.zeros_like(filtered_array, dtype=int)
    
    # Set to store unique categories
    unique_labels = set()
    
    # Iterate through the nodes in the graph
    for idx, node in enumerate(G.nodes):
        if attribute in G.nodes[node]:  # Ensure the node has the specified attribute
            category = G.nodes[node][attribute]
            radius = G.nodes[node]['radius']  # Use the node's radius attribute

            # Get the surface voxels of the ellipsoid
            surface_voxels = get_ellipsoid_surface(radius, scaled_voxel_size)

            # Adjust the node coordinates to match the anisotropic space
            adjusted_node = np.array([int(coord / scale) for coord, scale in zip(node, scaled_voxel_size)])
            adjusted_surface_voxels = (surface_voxels + adjusted_node).astype(int)

            # Set the surface voxels in category_indices using advanced indexing
            z_coords, y_coords, x_coords = adjusted_surface_voxels.T
            category_indices[z_coords, y_coords, x_coords] = category
            unique_labels.add(category)

    # Create a mask for the foreground voxels in category_indices
    foreground_mask = category_indices != 0

    # Compute distance transform from foreground to non-foreground regions
    _, indices = ndi.distance_transform_cdt(~foreground_mask, return_indices=True)

    # Map each non-zero voxel to the nearest category using the indices
    non_zero_mask = filtered_array != 0
    filtered_array[non_zero_mask] = category_indices[tuple(indices[:, non_zero_mask])]

    # Extract unique labels present in the segmented volume
    skel_label = sorted(unique_labels)

    return filtered_array, skel_label

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
