# src/main.py
import argparse
import numpy as np
from data_loader import load_hdf5, load_existing_skeletons
from graph_segmentation import (
    skeletonize_volume, full_graph_generation, get_branch_points, get_end_points,
    get_neighbor_counts, simplified_graph_generation, plot_elbow_curve, 
    cluster_radius, relabel_graph_with_branches, label_branch_points, calculate_branching_angles, 
    segment_volume
)
from data_saver import (
    save_skeleton_to_file, save_segmented_volume, save_stats, graph2video
)
from visualization import (
    visualize_skeleton, visualize_radii, visualize_skeleton_colored,
    plot_all_paths_radii
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
            
    # Generate the full graph
    G = full_graph_generation(skeletons)

    # Get branch points and end points
    branch_points = get_branch_points(G)
    end_points = get_end_points(G)
    neighbor_counts = get_neighbor_counts(G, branch_points)
    
    if args.debug:
        print("Number of branch points:", len(branch_points))
        print("Number of end points:", len(end_points))
    
    # Generate the simplified graph
    simplified_G, unique_paths, medians, means = simplified_graph_generation(G, branch_points, end_points)

    # Plot the elbow curve to determine the optimal number of clusters
    plot_elbow_curve(medians)
    optimal_clusters = int(input("Enter the optimal number of clusters: "))

    # Cluster the medians
    labels = cluster_radius(medians, optimal_clusters)

    # Relabel the graph and get branch details and total length
    G, branch_info, total_length = relabel_graph_with_branches(G, unique_paths, labels, medians, means)
    
    G = label_branch_points(G,branch_points)
    
    branching_angles = calculate_branching_angles(G, branch_points)
    
    # Save the stats
    save_stats(args.stats_output_path, branch_points, end_points, neighbor_counts, branch_info, total_length, branching_angles)

    # Segment the volume
    segmented_volume, skel_label = segment_volume(filtered_array, G, args.voxel_size, attribute=args.segmentation_attribute)

    # Save the segmented volume
    save_segmented_volume(segmented_volume, skel_label, args.output_file)
    
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
    parser.add_argument('--segmentation_attribute', type=str, default='label', help="Attribute to segment and visualize by")
    
    #Visualization parameters
    parser.add_argument('--scale_factor', type=int, default=10, help="Scale factor for visualization")
    parser.add_argument('--visualize_skeleton', action='store_true', help="Whether to visualize the skeleton")
    parser.add_argument('--visualize_radii', action='store_true', help="Whether to visualize the distribution of radii")
    parser.add_argument('--visualize_skeleton_colored', action='store_true', help="Whether to visualize the skeleton colored by attribute")
    parser.add_argument('--visualize_paths_radii', action='store_true', help="Whether to visualize the radii of paths")
    
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
            output_file = 'label_segmented_macaque_vessels.h5'  # Name of the output HDF5 file
            voxel_size = (320, 256, 256)  # Voxel sizes in z, y, x order
            
            generate_new_skeleton = False  # Set to False to use existing skeletons
            teasar_params = '{"scale": 1.5, "const": 42000, "pdrf_scale": 100000, "pdrf_exponent": 4, "soma_acceptance_threshold": 3500, "soma_detection_threshold": 750, "soma_invalidation_const": 300, "soma_invalidation_scale": 2, "max_paths": 300}'  # JSON string of TEASAR parameters
            existing_skeletons_path = None if generate_new_skeleton else r"C:\Users\14132\Desktop\Vessel2Graph\src\macaque_original_isotropy_skeleton.npz"
            
            save_skeleton = True
            skeleton_output_path = "macaque_original_isotropy_skeleton.npz"
            
            target_labels = [1]  # Labels to keep in the array
            segmentation_attribute = 'label'  # Attribute to segment and visualize by
            
            scale_factor = 1920  # Scale factor for visualization
            visualize_skeleton = False  # Whether to visualize the skeleton
            visualize_radii = False  # Whether to visualize the distribution of radii
            visualize_skeleton_colored = False  # Whether to visualize the skeleton colored by attribute
            visualize_paths_radii = False  # Whether to visualize the radii of paths
            
            create_video = False  # Whether to create a video
            num_rotations = 3  # Number of full rotations for video
            fps = 10  # Frames per second for video
            video_graph_type = 'simplified'  # Type of graph to use for the video ('full' or 'simplified')
            video_output_file = 'simplified_macaque.mp4'  # Name of the output video file (must end with .avi or .mp4)
            
            stats_output_path = 'macaque_stats'
            
            debug = False
            
        args = Args()
    main(args)
