# Vessel2Graph

Vessel2Graph is a project for skeletonizing and visualizing vessel data from 3D volumes. The project includes scripts for loading data, preprocessing, graph segmentation, visualization, and creating videos of the segmented vessels.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
  - [main.py](#mainpy)
  - [data_processing.py](#data_processingpy)
  - [graph_segmentation.py](#graph_segmentationpy)
  - [visualization.py](#visualizationpy)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Vessel2Graph.git
    cd Vessel2Graph
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Execution

To run the script from the command line:

1. **Prepare your data**:
    - Ensure your input data is in HDF5 format.

2. **Run the main script**:
    ```bash
    python src/main.py input_file.h5 dataset_name --output_file segmented_vessels.h5 --voxel_size 1.0 1.0 1.0 --teasar_params '{"scale": 1.5, "const": 600, "pdrf_scale": 100000, "pdrf_exponent": 4, "soma_acceptance_threshold": 3500, "soma_detection_threshold": 750, "soma_invalidation_const": 300, "soma_invalidation_scale": 2, "max_paths": 300}' --target_labels 1 --segmentation_attribute label --scale_factor 10 --num_rotations 3 --fps 10 --visualize_skeleton --visualize_radii --visualize_skeleton_colored --visualize_paths_radii --create_video --video_graph_type full --video_output_file output_video.avi
    ```

### IDE Execution

To run the script in an IDE like Spyder or PyCharm:

1. **Prepare your data**:
    - Ensure your input data is in HDF5 format.

2. **Run the main script**:
    - Open `main.py` in your IDE.
    - Manually set the values for the `Args` class within the script.
    - Execute the script.

Example of the `Args` class in `main.py`:
```python
if __name__ == "__main__":
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
```

## Project Structure

- `src/`: Contains the main scripts for processing and visualization.
- `tests/`: Contains test scripts.
- `README.md`: This file.
- `requirements.txt`: Python dependencies.

## Scripts

### main.py

The main script orchestrates the entire pipeline from data loading to visualization.

- **Arguments**:
  - `input_file`: Path to the input HDF5 file.
  - `dataset_name`: Name of the dataset in the HDF5 file.
  - `output_file`: Name of the output HDF5 file.
  - `voxel_size`: Voxel sizes in z, y, x order.
  - `teasar_params`: JSON string of TEASAR parameters.
  - `target_labels`: Labels to keep in the array.
  - `segmentation_attribute`: Attribute to segment and visualize by.
  - `scale_factor`: Scale factor for visualization.
  - `num_rotations`: Number of full rotations for video.
  - `fps`: Frames per second for video.
  - `visualize_skeleton`: Whether to visualize the skeleton.
  - `visualize_radii`: Whether to visualize the distribution of radii.
  - `visualize_skeleton_colored`: Whether to visualize the skeleton colored by attribute.
  - `visualize_paths_radii`: Whether to visualize the radii of paths.
  - `create_video`: Whether to create a video.
  - `video_graph_type`: Type of graph to use for the video (`full` or `simplified`).
  - `video_output_file`: Name of the output video file (must end with .avi or .mp4).

### data_processing.py

Contains functions for loading data, filtering labels, and scaling voxel sizes.

### graph_segmentation.py

Contains functions for skeletonizing the volume, generating graphs, identifying branch points, segmenting the volume, and clustering.

### visualization.py

Contains functions for visualizing the skeleton, radii, colored skeletons, path radii histograms, and creating videos from graphs.
