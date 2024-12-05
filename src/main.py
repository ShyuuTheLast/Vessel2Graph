# src/main.py
import argparse
import numpy as np
from data_loader import load_hdf5, load_existing_skeletons
from graph_segmentation import (
    skeletonize_volume, full_graph_generation, get_branch_points, get_end_points,
    merge_graph_components, get_neighbor_counts, simplified_graph_generation, plot_elbow_curve, 
    cluster_radius, relabel_graph_with_branches, relabel_small_branches_near_big_ones, label_branch_points, calculate_branching_angles,
    calculate_distance_from_largest, propagate_distances_to_original_graph, remove_small_components_cc3d, 
    segment_volume, skeleton_in_array
)
from data_saver import (
    save_skeleton_to_file, save_segmented_volume, save_stats, graph2video, save_paths_as_pickle
)
from visualization import (
    visualize_skeleton, visualize_radii, visualize_skeleton_colored,
    plot_all_paths_radii, visualize_volume_heatmap
)

def main():
    input_file = r"C:\Users\14132\Desktop\BC Research Internship\Vessel2Graph\macaque_mpi_176-240nm_bv_gt_cc.h5"  # Path to the input HDF5 file
    dataset_name = 'main'  # Name of the dataset in the HDF5 file
    output_file = 'down_branch_macaque_vessels.h5'  # Name of the output HDF5 file
    voxel_size = (320, 256, 256)  # Voxel sizes in z, y, x order

    generate_new_skeleton = False  # Set to False to use existing skeletons
    teasar_params = '{"scale": 1.5, "const": 30000, "pdrf_scale": 100000, "pdrf_exponent": 4, "soma_acceptance_threshold": 3500, "soma_detection_threshold": 750, "soma_invalidation_const": 300, "soma_invalidation_scale": 2, "max_paths": 300}'  # JSON string of TEASAR parameters
    existing_skeletons_path = None if generate_new_skeleton else r"C:\Users\14132\Desktop\Vessel2Graph\src\down_macaque_skel.npz"
    save_skeleton = True
    skeleton_output_path = "macaque_original_isotropy_skeleton.npz"

    target_labels = [1]  # Labels to keep in the array
    relabel_nodes = True

    save_paths_list = True  # Whether to save the nodes of skeleton as list of lists divided by branches
    paths_list_name = "macaque_branches.pkl"
    save_skeleton_as_array = True  # Whether to save skeleton as points in a volume
    skeleton_array_name = "macaque_skel_as_array"

    do_segment = False;  # Whether to segment and save original volume
    segmentation_attribute = 'branch'  # Current options: branch, label, radius, dist_from_largest

    scale_factor = 1920  # Scale factor for visualization
    visualize_skeleton = False  # Whether to visualize the skeleton
    visualize_radii = False  # Whether to visualize the distribution of radii
    visualize_skeleton_colored = False  # Whether to visualize the skeleton colored by attribute
    visualize_paths_radii = False  # Whether to visualize the radii of paths
    visualize_heatmap = False  # Whether to visualize heatmap based off radii

    create_video = False  # Whether to create a video
    num_rotations = 3  # Number of full rotations for video
    fps = 10  # Frames per second for video
    video_graph_type = 'simplified'  # Type of graph to use for the video ('full' or 'simplified')
    video_output_file = 'simplified_macaque.mp4'  # Name of the output video file (must end with .avi or .mp4)

    stats_output_path = 'macaque_stats'

    debug = False

    # Load and preprocess the data
    filtered_array = load_hdf5(input_file, dataset_name, target_labels=target_labels)
    
    if debug:
        print("Unique values in the array:", np.unique(filtered_array))
    
    # Handle skeleton generation or loading existing skeletons
    if generate_new_skeleton:
        # Skeletonize the volume
        teasar_params = eval(teasar_params) if teasar_params else {}
        skeletons = skeletonize_volume(filtered_array, teasar_params, anisotropy=voxel_size)

        # Optionally save the generated skeleton
        if save_skeleton:
            save_skeleton_to_file(skeletons, skeleton_output_path)
    else:
        # Load existing skeletons from the specified path
        skeletons = load_existing_skeletons(existing_skeletons_path)

    if debug:
        for key, value in skeletons.items():
            print(f"Key: {key}, Number of vertices: {value.vertices.shape[0]}")
            
    # Generate the full graph from the skeletons
    G = full_graph_generation(skeletons)
    
    # Get branch points and end points in the graph
    branch_points = get_branch_points(G)
    end_points = get_end_points(G)
    
    G, end_points = merge_graph_components(G, end_points)
    
    # Get the count of neighbors for each branch point
    neighbor_counts = get_neighbor_counts(G, branch_points)
    
    # Debug information: Print the number of branch and end points if debugging is enabled
    if debug:
        print("Number of branch points:", len(branch_points))
        print("Number of end points:", len(end_points))
    
    # Generate a simplified version of the graph along with path details
    simplified_G, unique_paths, medians, means, vertices_by_branch = simplified_graph_generation(G, branch_points, end_points, voxel_size)
    
    if save_paths_list:
        save_paths_as_pickle(vertices_by_branch, paths_list_name)
    
    # Plot the elbow curve to help determine the optimal number of clusters for k-means
    plot_elbow_curve(medians)
    optimal_clusters = int(input("Enter the optimal number of clusters: "))
    #optimal_clusters = 2
    
    # Cluster the medians of the branches to classify them into different groups
    labels, largest_cluster_label, second_largest_label = cluster_radius(medians, optimal_clusters)
    
    # Relabel the original graph with branch indices, calculate branch details and the total length of the skeleton
    G, branch_info, total_length = relabel_graph_with_branches(G, unique_paths, labels, medians, means)
    
    # Calculate the branching angles and create a graph to represent branch connectivity
    branching_angles, branch_connectivity_graph = calculate_branching_angles(G, branch_points)
    
    if relabel_nodes:
        # Relabel nodes in small branches that are neighboring large branches
        G, branch_connectivity_graph = relabel_small_branches_near_big_ones(G, branch_connectivity_graph, largest_cluster_label, unique_paths, branch_points)
    
    # Label the branch points in the graph with the correct branch labels
    G = label_branch_points(G, branch_points, largest_cluster_label)

    #export list of branches with largest cluster label
    if args.segmentation_attribute == "label":
        large_branches = []

        for branch in branch_connectivity_graph.nodes:
            if branch_connectivity_graph.nodes[branch]['label'] and branch_connectivity_graph.nodes[branch]['label'] == largest_cluster_label:
                large_branches.append(branch)
        large_branches.sort()

        with open('large_branches.txt','w') as file:
            for branch_id in large_branches:
                file.write(f"{branch_id}\n")
    
    if segmentation_attribute == "dist_from_largest":
        # Calculate the distance from each branch to the nearest branch in the largest cluster
        distances = calculate_distance_from_largest(branch_connectivity_graph, largest_cluster_label)
        
        # Propagate these distances back to the original graph
        G = propagate_distances_to_original_graph(G, branch_connectivity_graph, distances)
    
    if save_skeleton_as_array:
        skeleton_array = skeleton_in_array(filtered_array, G, voxel_size)
        save_segmented_volume(skeleton_array,skeleton_array_name)
    
    # Save all the calculated statistics (branch points, end points, etc.) to a file
    save_stats(stats_output_path, branch_points, end_points, neighbor_counts, branch_info, total_length, branching_angles)
    
    if do_segment:
        # Segment the volume using the graph's node attributes (e.g., label or radius)
        segmented_volume, foreground_mask = segment_volume(filtered_array, G, voxel_size, attribute=segmentation_attribute)
        
        if debug and segmentation_attribute == "label":
            save_segmented_volume(np.where(foreground_mask, 2, filtered_array),"foreground_mask")
        
        if segmentation_attribute == "label":
            # Perform connected component analysis and relabel dust
            segmented_volume = remove_small_components_cc3d(segmented_volume, largest_cluster_label, second_largest_label, threshold_ratio=0.01, connectivity=26)
        
        # Save the segmented volume to an HDF5 file
        save_segmented_volume(segmented_volume, output_file)
    
    # Visualizations
    if visualize_skeleton:
        visualize_skeleton(skeletons, scale_factor=scale_factor)
        # Note: The script will not proceed unless the Mayavi window is closed
    
    if visualize_radii:
        visualize_radii(skeletons)
    
    if visualize_skeleton_colored:
        visualize_skeleton_colored(G, attribute=segmentation_attribute, scale_factor=scale_factor)
        # Note: The script will not proceed unless the Mayavi window is closed

    if visualize_paths_radii:
        plot_all_paths_radii(unique_paths)
    
    if visualize_heatmap:
        if segmentation_attribute == 'radius':
            visualize_volume_heatmap(segmented_volume)
        else:
            print('Heatmap visualization is available only for volume segmented by radius')
    
    # Create a video of the graph if specified
    if create_video:
        if video_graph_type == 'full':
            graph2video(G, video_output_file, num_rotations=num_rotations, fps=fps)
        elif video_graph_type == 'simplified':
            graph2video(simplified_G, video_output_file, num_rotations=num_rotations, fps=fps)
        
if __name__ == "__main__":
    main()
