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


def main(args):
    # Load and preprocess the data
    filtered_array = load_hdf5(args.input_file, args.dataset_name, target_labels=args.target_labels)
    
    if args.debug:
        print("Unique values in the array:", np.unique(filtered_array))
    
    # Handle skeleton generation or loading existing skeletons
    if args.generate_new_skeleton:
        # Skeletonize the volume
        teasar_params = eval(args.teasar_params) if args.teasar_params else {}
        skeletons = skeletonize_volume(filtered_array, teasar_params, anisotropy=args.voxel_size)

        # Optionally save the generated skeleton
        if args.save_skeleton:
            save_skeleton_to_file(skeletons, args.skeleton_output_path)
    else:
        # Load existing skeletons from the specified path
        skeletons = load_existing_skeletons(args.existing_skeletons_path)

    if args.debug:
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
    if args.debug:
        print("Number of branch points:", len(branch_points))
        print("Number of end points:", len(end_points))
    
    # Generate a simplified version of the graph along with path details
    simplified_G, unique_paths, medians, means, vertices_by_branch = simplified_graph_generation(G, branch_points, end_points, args.voxel_size)
    
    if args.save_paths_list:
        save_paths_as_pickle(vertices_by_branch, args.paths_list_name)
    
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
    
    if args.relabel_nodes:
        # Relabel nodes in small branches that are neighboring large branches
        G, branch_connectivity_graph = relabel_small_branches_near_big_ones(G, branch_connectivity_graph, largest_cluster_label, unique_paths, branch_points)
    
    # Label the branch points in the graph with the correct branch labels
    G = label_branch_points(G, branch_points, largest_cluster_label)
    
    if args.segmentation_attribute == "dist_from_largest":
        # Calculate the distance from each branch to the nearest branch in the largest cluster
        distances = calculate_distance_from_largest(branch_connectivity_graph, largest_cluster_label)
        
        # Propagate these distances back to the original graph
        G = propagate_distances_to_original_graph(G, branch_connectivity_graph, distances)
    
    if args.save_skeleton_as_array:
        skeleton_array = skeleton_in_array(filtered_array, G, args.voxel_size)
        save_segmented_volume(skeleton_array,args.skeleton_array_name)
    
    # Save all the calculated statistics (branch points, end points, etc.) to a file
    save_stats(args.stats_output_path, branch_points, end_points, neighbor_counts, branch_info, total_length, branching_angles)
    
    if args.do_segment:
        # Segment the volume using the graph's node attributes (e.g., label or radius)
        segmented_volume, foreground_mask = segment_volume(filtered_array, G, args.voxel_size, attribute=args.segmentation_attribute)
        
        if args.debug and args.segmentation_attribute == "label":
            save_segmented_volume(np.where(foreground_mask, 2, filtered_array),"foreground_mask")
        
        if args.segmentation_attribute == "label":
            # Perform connected component analysis and relabel dust
            segmented_volume = remove_small_components_cc3d(segmented_volume, largest_cluster_label, second_largest_label, threshold_ratio=0.01, connectivity=26)
        
        # Save the segmented volume to an HDF5 file
        save_segmented_volume(segmented_volume, args.output_file)
    
    # Visualizations
    if args.visualize_skeleton:
        visualize_skeleton(skeletons, scale_factor=args.scale_factor)
        # Note: The script will not proceed unless the Mayavi window is closed
    
    if args.visualize_radii:
        visualize_radii(skeletons)
    
    if args.visualize_skeleton_colored:
        visualize_skeleton_colored(G, attribute=args.segmentation_attribute, scale_factor=args.scale_factor)
        # Note: The script will not proceed unless the Mayavi window is closed

    if args.visualize_paths_radii:
        plot_all_paths_radii(unique_paths)
    
    if args.visualize_heatmap:
        if args.segmentation_attribute == 'radius':
            visualize_volume_heatmap(segmented_volume)
        else:
            print('Heatmap visualization is available only for volume segmented by radius')
    
    # Create a video of the graph if specified
    if args.create_video:
        if args.video_graph_type == 'full':
            graph2video(G, args.video_output_file, num_rotations=args.num_rotations, fps=args.fps)
        elif args.video_graph_type == 'simplified':
            graph2video(simplified_G, args.video_output_file, num_rotations=args.num_rotations, fps=args.fps)
        
def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for processing and visualizing skeletonized blood vessel volumes.")
    
    # General parameters
    parser.add_argument('input_file', type=str, help="Path to the input HDF5 file")
    parser.add_argument('dataset_name', type=str, help="Name of the dataset in the HDF5 file")
    parser.add_argument('--output_file', type=str, default='segmented_vessels.h5', help="Name of the output HDF5 file")
    
    # Data processing parameters
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Voxel sizes in z, y, x order")
    
    # New parameters for skeleton generation
    parser.add_argument('--generate_new_skeleton', action='store_true', help="Flag to generate a new skeleton using Kimimaro")
    parser.add_argument('--teasar_params', type=str, help="JSON string of TEASAR parameters (required if generate_new_skeleton is true)")
    parser.add_argument('--existing_skeletons_path', type=str, help="Path to existing skeletons (required if generate_new_skeleton is false)")
    
    # New parameters for saving skeleton
    parser.add_argument('--save_skeleton', action='store_true', help="Flag to save the generated skeleton")
    parser.add_argument('--skeleton_output_path', type=str, help="Path to save the skeleton (required if save_skeleton is true)")
    
    # Segmentation and 
    parser.add_argument('--target_labels', type=int, nargs='+', default=[1], help="Labels to keep in the array")
    parser.add_argument('--relabel_nodes', action='store_true', help="Whether to relabel nodes of small branches near large branches to big")
    parser.add_argument('--segmentation_attribute', type=str, default='label', help="Attribute to segment and visualize by")
    
    #Visualization parameters
    parser.add_argument('--scale_factor', type=int, default=10, help="Scale factor for visualization")
    parser.add_argument('--visualize_skeleton', action='store_true', help="Whether to visualize the skeleton")
    parser.add_argument('--visualize_radii', action='store_true', help="Whether to visualize the distribution of radii")
    parser.add_argument('--visualize_skeleton_colored', action='store_true', help="Whether to visualize the skeleton colored by attribute")
    parser.add_argument('--visualize_paths_radii', action='store_true', help="Whether to visualize the radii of paths")
    parser.add_argument('--visualize_heatmap', action='store_true', help="Whether to visualize the heatmap representing the volume segmented by node radius")

    # Video generation parameters
    parser.add_argument('--create_video', action='store_true', help="Whether to create a video")
    parser.add_argument('--num_rotations', type=int, default=3, help="Number of full rotations for video")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second for video")
    parser.add_argument('--video_graph_type', type=str, choices=['full', 'simplified'], default='simplified', help="Type of graph to use for the video ('full' or 'simplified')")
    parser.add_argument('--video_output_file', type=str, default='output_video.avi', help="Name of the output video file (must end with .avi or .mp4)")
    
   # Parameter for stats output path
    parser.add_argument('--stats_output_path', type=str, help="Path to save the statistics output")
    
    parser.add_argument('--debug', action='store_true', help="Whether to print various debug information")
    
    args = parser.parse_args()

    # Validate the arguments
    if args.generate_new_skeleton:
        if not args.teasar_params:
            parser.error("--teasar_params is required when --generate_new_skeleton is true")
    else:
        if not args.existing_skeletons_path:
            parser.error("--existing_skeletons_path is required when --generate_new_skeleton is false")
    
    if args.save_skeleton:
        if not args.skeleton_output_path:
            parser.error("--skeleton_output_path is required when --save_skeleton is true")
    
    if args.create_video:
        # Ensure that all video-related parameters are provided
        if not (args.scale_factor and args.num_rotations and args.fps and args.video_output_file):
            parser.error("--create_video requires scale_factor, num_rotations, fps, and video_output_file")
    
    return args

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        # Manually set args for testing in an IDE
        class Args:
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
            
            save_paths_list = True # Whether to save the nodes of skeleton as list of lists divided by branches
            paths_list_name = "macaque_branches.pkl"
            save_skeleton_as_array = True #Whether to save skeleton as points in a volume
            skeleton_array_name = "macaque_skel_as_array"
            
            do_segment = False; # Whether to segment and save original volume
            segmentation_attribute = 'branch'  # Current options: branch, label, radius, dist_from_largest
            
            scale_factor = 1920  # Scale factor for visualization
            visualize_skeleton = False  # Whether to visualize the skeleton
            visualize_radii = False  # Whether to visualize the distribution of radii
            visualize_skeleton_colored = False  # Whether to visualize the skeleton colored by attribute
            visualize_paths_radii = False  # Whether to visualize the radii of paths
            visualize_heatmap = False # Whether to visualize heatmap based off radii
            
            create_video = False  # Whether to create a video
            num_rotations = 3  # Number of full rotations for video
            fps = 10  # Frames per second for video
            video_graph_type = 'simplified'  # Type of graph to use for the video ('full' or 'simplified')
            video_output_file = 'simplified_macaque.mp4'  # Name of the output video file (must end with .avi or .mp4)
            
            stats_output_path = 'macaque_stats'
            
            debug = False
            
        args = Args()
    main(args)
