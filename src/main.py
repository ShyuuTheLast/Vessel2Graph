# src/main.py
import argparse
from data_loader import load_hdf5, scale_isotropy
from graph_segmentation import (
    skeletonize_volume, full_graph_generation, get_branch_points, get_end_points,
    simplified_graph_generation, get_median_radii, plot_elbow_curve, cluster_medians,
    relabel_graph_with_branches, segment_volume, save_segmented_volume, scale_graph
)
from visualization import (
    visualize_skeleton, visualize_radii, visualize_skeleton_colored,
    plot_all_paths_radii, graph2video
)

def main(args):
    # Load and preprocess the data
    filtered_array = load_hdf5(args.input_file, args.dataset_name, target_labels=args.target_labels)
    scaled_voxel_size, common_factor = scale_isotropy(args.voxel_size)

    # Skeletonize the volume
    teasar_params = eval(args.teasar_params) if args.teasar_params else {}
    skeletons = skeletonize_volume(filtered_array, teasar_params, anisotropy=scaled_voxel_size)

    # Generate the full graph
    G = full_graph_generation(skeletons)

    # Get branch points and end points
    branch_points = get_branch_points(G)
    end_points = get_end_points(G)

    # Generate the simplified graph
    simplified_G, unique_paths = simplified_graph_generation(G, branch_points, end_points)

    # Get median radii of each branch
    medians = get_median_radii(unique_paths)

    # Plot the elbow curve to determine the optimal number of clusters
    plot_elbow_curve(medians)
    optimal_clusters = int(input("Enter the optimal number of clusters: "))

    # Cluster the medians
    labels = cluster_medians(medians, optimal_clusters)

    # Relabel the graph with branches and clusters
    G = relabel_graph_with_branches(G, unique_paths, labels)

    # Segment the volume
    segmented_volume, skel_label = segment_volume(filtered_array, G, scaled_voxel_size, attribute=args.segmentation_attribute)

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
            scaled_G = scale_graph(G, common_factor)
        elif args.video_graph_type == 'simplified':
            scaled_G = scale_graph(simplified_G, common_factor)

        graph2video(scaled_G, args.video_output_file, num_rotations=args.num_rotations, fps=args.fps)

def parse_args():
    parser = argparse.ArgumentParser(description="Skeletonize and visualize vessel data from 3D volumes.")
    parser.add_argument('input_file', type=str, help="Path to the input HDF5 file")
    parser.add_argument('dataset_name', type=str, help="Name of the dataset in the HDF5 file")
    parser.add_argument('--output_file', type=str, default='segmented_vessels.h5', help="Name of the output HDF5 file")
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Voxel sizes in z, y, x order")
    parser.add_argument('--teasar_params', type=str, default='{"scale": 1.5, "const": 600, "pdrf_scale": 100000, "pdrf_exponent": 4, "soma_acceptance_threshold": 3500, "soma_detection_threshold": 750, "soma_invalidation_const": 300, "soma_invalidation_scale": 2, "max_paths": 300}', help="JSON string of TEASAR parameters")
    parser.add_argument('--target_labels', type=int, nargs='+', default=[1], help="Labels to keep in the array")
    parser.add_argument('--segmentation_attribute', type=str, default='label', help="Attribute to segment and visualize by")
    parser.add_argument('--scale_factor', type=int, default=10, help="Scale factor for visualization")
    parser.add_argument('--num_rotations', type=int, default=3, help="Number of full rotations for video")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second for video")
    parser.add_argument('--visualize_skeleton', action='store_true', help="Whether to visualize the skeleton")
    parser.add_argument('--visualize_radii', action='store_true', help="Whether to visualize the distribution of radii")
    parser.add_argument('--visualize_skeleton_colored', action='store_true', help="Whether to visualize the skeleton colored by attribute")
    parser.add_argument('--visualize_paths_radii', action='store_true', help="Whether to visualize the radii of paths")
    parser.add_argument('--create_video', action='store_true', help="Whether to create a video")
    parser.add_argument('--video_graph_type', type=str, choices=['full', 'simplified'], default='simplified', help="Type of graph to use for the video ('full' or 'simplified')")
    parser.add_argument('--video_output_file', type=str, default='output_video.avi', help="Name of the output video file (must end with .avi or .mp4)")
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        # Manually set args for testing in an IDE
        class Args:
            input_file = r"C:\Users\14132\Desktop\BC Research Internship\Vessel2Graph\macaque_mpi_176-240nm_bv_gt_cc.h5"  # Path to the input HDF5 file
            dataset_name = 'main'  # Name of the dataset in the HDF5 file
            output_file = 'segmented_macaque_vessels.h5'  # Name of the output HDF5 file
            voxel_size = (320, 256, 256)  # Voxel sizes in z, y, x order
            teasar_params = '{"scale": 1.5, "const": 600, "pdrf_scale": 100000, "pdrf_exponent": 4, "soma_acceptance_threshold": 3500, "soma_detection_threshold": 750, "soma_invalidation_const": 300, "soma_invalidation_scale": 2, "max_paths": 300}'  # JSON string of TEASAR parameters
            target_labels = [1]  # Labels to keep in the array
            segmentation_attribute = 'label'  # Attribute to segment and visualize by
            scale_factor = 30  # Scale factor for visualization
            num_rotations = 3  # Number of full rotations for video
            fps = 10  # Frames per second for video
            visualize_skeleton = True  # Whether to visualize the skeleton
            visualize_radii = True  # Whether to visualize the distribution of radii
            visualize_skeleton_colored = True  # Whether to visualize the skeleton colored by attribute
            visualize_paths_radii = True  # Whether to visualize the radii of paths
            create_video = True  # Whether to create a video
            video_graph_type = 'simplified'  # Type of graph to use for the video ('full' or 'simplified')
            video_output_file = 'simplified_macaque.mp4'  # Name of the output video file (must end with .avi or .mp4)

        args = Args()
    main(args)
