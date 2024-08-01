# src/data_loader.py
import h5py
import numpy as np

def load_hdf5(file_path, dataset_name, target_labels=[1]):
    """
    Load data from an HDF5 file and filter it to keep only the specified labels.

    Parameters:
    - file_path (str): Path to the HDF5 file.
    - dataset_name (str): Name of the dataset to load.
    - target_labels (list of int): Labels to keep in the array (default is [1]).

    Returns:
    - np.ndarray: Filtered data array.
    """
    with h5py.File(file_path, 'r') as h5file:
        data_array = np.array(h5file[dataset_name])
    return filter_label(data_array, target_labels)

def filter_label(data_array, target_labels):
    """
    Filter the data array to keep only the specified labels.

    Parameters:
    - data_array (np.ndarray): Input data array.
    - target_labels (list of int): Labels to keep in the array (default is [1]).

    Returns:
    - np.ndarray: Filtered data array.
    """
    if isinstance(target_labels, int):
        target_labels = [target_labels]
    mask = np.isin(data_array, target_labels)
    return np.where(mask, data_array, 0)

def load_existing_skeletons(file_path):
    """
    Load skeletons from a .npz file.

    Parameters:
    - file_path (str): The path to the .npz file containing the skeletons.

    Returns:
    - dict: A dictionary of skeleton objects, where each key is an ID and each value is a skeleton.
    """
    loaded_data = np.load(file_path, allow_pickle=True)
    skeletons = loaded_data['skeletons'].item()
    return skeletons