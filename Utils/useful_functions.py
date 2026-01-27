import os
import sys
import json
import torch
import pickle
import random
import warnings
import numpy as np

from Utils.metric_utils.estimation import *
from Utils.metric_utils.distribution import *


def print_(*args, file=None):
    """
    Print function with automatic flushing.
    
    Prints the given arguments to the specified file (or stdout by default)
    and ensures the output is immediately flushed to guarantee it appears
    in real-time, especially useful for logging during long-running processes.
    
    Args:
        *args: Variable length argument list to print
        file: File object to print to (default: sys.stdout)
    """
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()


def check_and_create_dir(dir_path):
    """
    Check if a directory exists and create it if it doesn't.
    
    This function checks whether the specified directory path exists,
    and creates the directory (including any necessary parent directories)
    if it does not already exist.
    
    Args:
        dir_path (str): Path to the directory to check/create
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random.
    
    This ensures reproducible results across runs by setting the same
    initial seed for all relevant random number generators.
    
    Args:
        seed (int, optional): The integer value seed for global random state.
                             If None, no seeding is performed.
        cudnn_deterministic (bool): Whether to set CuDNN to deterministic mode.
                                   This may impact performance but ensures
                                   reproducible results on GPU. Default: False.
                                   
    Note:
        When cudnn_deterministic is True, it may significantly slow down training
        and could cause issues when resuming from checkpoints.
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic

        if cudnn_deterministic:
            warnings.warn('You have chosen to enable CuDNN deterministic mode. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')


def save_to_json(data, data_path='data.json'):
    """
    Save data to a JSON file.
    
    Args:
        data: The data to be saved. Can be any JSON serializable Python object.
        data_path: The path to the file where data will be saved. Defaults to 'data.json'.
        
    Returns:
        None
    """
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_from_json(data_path='data.json'):
    """
    Load data from a JSON file.
    
    Args:
        data_path: The path to the JSON file to load data from. Defaults to 'data.json'.
        
    Returns:
        The data loaded from the JSON file. The data type depends on the content
        of the JSON file, typically dict, list, str, int, float, or bool.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def merge_results_by_name(A1, names1, A2, names2):
    """
    Merge two arrays based on their associated names by reordering and concatenating along the first dimension.
    
    This function takes two 3D arrays and their corresponding name lists, reorders the arrays according to 
    the names, and concatenates them along the first axis (k-dimension). It ensures that both name lists 
    contain the same elements and maintains consistent ordering.
    
    Args:
        A1 (array-like): First input array with shape (k1, num_sketches, num_keys)
        names1 (list): List of names corresponding to the sketches in A1 (length = num_sketches)
        A2 (array-like): Second input array with shape (k2, num_sketches, num_keys)
        names2 (list): List of names corresponding to the sketches in A2 (length = num_sketches)
        
    Returns:
        tuple: A tuple containing:
            - merged (numpy.ndarray): Concatenated array with shape (k1+k2, num_sketches, num_keys)
            - ordered_names (list): Ordered list of names used for reordering
        
    Raises:
        AssertionError: If arrays don't have 3 dimensions, shapes don't align properly, or name lists differ
    """
    A1 = np.asarray(A1)
    A2 = np.asarray(A2)

    # -------- shape checks --------
    assert A1.ndim == 3 and A2.ndim == 3
    k1, num_sketches, num_keys = A1.shape
    k2, num_sketches2, num_keys2 = A2.shape

    assert num_sketches == num_sketches2
    assert num_keys == num_keys2
    assert len(names1) == num_sketches
    assert len(names2) == num_sketches

    # -------- name consistency --------
    set1, set2 = set(names1), set(names2)
    assert set1 == set2, "The two name lists must contain the same set of elements"

    ordered_names = names1

    # name -> index
    map1 = {name: i for i, name in enumerate(names1)}
    map2 = {name: i for i, name in enumerate(names2)}

    # -------- reorder sketch dimension --------
    out1 = np.empty_like(A1)
    out2 = np.empty_like(A2)

    for j, name in enumerate(ordered_names):
        out1[:, j, :] = A1[:, map1[name], :]
        out2[:, j, :] = A2[:, map2[name], :]

    # -------- concatenate along k dimension --------
    merged = np.concatenate([out1, out2], axis=0)

    return merged, ordered_names


def save_to_pickle(data: np.ndarray, names: list, data_path='.pkl'):
    """
    Save data and corresponding names to a pickle file.
    
    This function stores the given data array and names list as a dictionary
    in a pickle file. The data is stored with key "data" and names with key "names".
    
    Args:
        data (np.ndarray): The data array to be saved.
        names (list): The list of names corresponding to the data entries.
        data_path (str, optional): The path to the pickle file where data will be saved.
                                  Defaults to '.pkl'.
        
    Returns:
        None
    """
    save_dict = {
        "data": data,
        "names": names
    }
    with open(data_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(data_path='.pkl'):
    """
    Load data and names from a pickle file.
    
    Opens and reads a pickle file that contains a dictionary with 'data' and 'names' keys,
    then returns these values separately.
    
    Args:
        data_path (str): Path to the pickle file to load from. Defaults to '.pkl'.
        
    Returns:
        tuple: A tuple containing two elements:
            - The 'data' value from the pickle file
            - The 'names' value from the pickle file
    """
    with open(data_path, "rb") as f:
        load_dict = pickle.load(f)
    return load_dict["data"], load_dict["names"]


def get_evaluation_metrics(
    results: np.ndarray, 
    names: list, 
    has_aae: bool = True, 
    has_are: bool = True, 
    has_awe: bool = True,
    has_hhe: bool = True,
    has_eae: bool = True,
    has_wmrd: bool = True,
    logger = None
):
    """
    Compute and display various evaluation metrics for sketch algorithms.
    
    This function calculates multiple error metrics comparing predictions against ground truth,
    including Average Absolute Error (AAE), Average Relative Error (ARE), Average Weighted Error (AWE),
    Heavy Hitter Error (HHE), Entropy Absolute Error (EAE), and Weighted Mean Relative Difference (WMRD).
    Results are either logged using the provided logger or printed to stdout.
    
    Args:
        results (np.ndarray): A 2D array where each row represents results from a different method.
                             The ground truth ('gt') must be one of the rows.
        names (list): List of strings identifying each row in results. Must include 'gt' for ground truth.
        has_aae (bool): Whether to compute Average Absolute Error. Defaults to True.
        has_are (bool): Whether to compute Average Relative Error. Defaults to True.
        has_awe (bool): Whether to compute Average Weighted Error. Defaults to True.
        has_hhe (bool): Whether to compute Heavy Hitter Error. Defaults to True.
        has_eae (bool): Whether to compute Entropy Absolute Error. Defaults to True.
        has_wmrd (bool): Whether to compute Weighted Mean Relative Difference. Defaults to True.
        logger (Logger, optional): Logger instance for output. If None, uses print() instead.
        
    Returns:
        tuple: A tuple containing:
            - return_dict (list): List of metric results in the order determined by which metrics were computed
            - metrics (list): List of metric names corresponding to the results in return_dict
            
    Raises:
        AssertionError: If 'gt' is not in names or if the length of names doesn't match results.shape[0]
    """
    assert 'gt' in names, "The ground truth must be included in the name list"
    assert len(names) == results.shape[0], f"The number of names {len(names)} must match the number of results {results.shape[0]}"
    
    # Create a mapping from names to indices for easy lookup
    map = {name: i for i, name in enumerate(names)}
    truth = results[map['gt'], :]
    
    # Initialize lists to store metric results
    aaes, ares, awes, hhes, eaes, wmrds = [], [], [], [], [], []

    # names.append('mean')
    # names.append('zero')

    # Calculate metrics for each method except ground truth
    for name in names:
        if name == 'gt': 
            continue
            
        if name == 'mean':
            predict = np.mean(truth) * np.ones_like(truth)
        elif name == 'zero':
            predict = np.zeros_like(truth)
        else:
            predict = results[map[name], :]

        # Compute various error metrics
        aae = compute_average_absolute_error(truth, predict) if has_aae else np.inf
        are = compute_average_relative_error(truth, predict) if has_are else np.inf
        awe = compute_average_weighted_error(truth, predict) if has_awe else np.inf
        hhe = compute_heavy_hitter_error(truth, predict)[-1] if has_hhe else np.inf
        eae = compute_entropy_absolute_error(truth, predict) if has_eae else np.inf
        wmrd = compute_weighted_mean_relative_difference(truth, predict) if has_wmrd else np.inf

        # Store results
        aaes.append(aae), ares.append(are), awes.append(awe), hhes.append(hhe), eaes.append(eae), wmrds.append(wmrd)

        # Format output string with all computed metrics
        output_str = f"{name}:"

        if has_aae:
            output_str += f" AAE={aae:.4f}"

        if has_are:
            output_str += f", ARE={are:.4f}"

        if has_awe:
            output_str += f", AWE={awe:.4f}"

        if has_hhe:
            output_str += f", HHE={hhe:.4f}"

        if has_eae:
            output_str += f", EAE={eae:.4f}"

        if has_wmrd:
            output_str += f", WMRD={wmrd:.4f}"
            
        # Log or print the results
        if logger is not None:
            logger.info(output_str)
        else:
            # print(output_str)
            pass

    # Prepare return values based on which metrics were computed
    return_dict, metrics = [], []

    if has_aae: 
        return_dict.append(aaes) 
        metrics.append('AAE')

    if has_are: 
        return_dict.append(ares) 
        metrics.append('ARE')

    if has_awe: 
        return_dict.append(awes) 
        metrics.append('AWE')

    if has_hhe: 
        return_dict.append(hhes) 
        metrics.append('HHE')

    if has_eae: 
        return_dict.append(eaes) 
        metrics.append('EAE')

    if has_wmrd: 
        return_dict.append(wmrds) 
        metrics.append('WMRD')

    return return_dict, metrics


def has_nan_or_not(tensor: torch.Tensor) -> bool:
    """
    Return True if the PyTorch tensor contains any NaN values.
    """
    return torch.isnan(tensor).any().item()


def load_results_pkls(dir_path):
    """
    Load multiple pickle files containing results from a directory into a dictionary.
    
    This function scans the specified directory for files with names following the pattern
    'results_*.pkl', loads each pickle file, and stores the contents in a dictionary where
    the key is extracted from the filename (the part between 'results_' and '.pkl').
    
    Args:
        dir_path (str): Path to the directory containing the pickle files to load
        
    Returns:
        dict: A dictionary mapping tags (extracted from filenames) to the loaded data
              from each pickle file
    """
    results_dict = {}

    # Iterate through all files in the directory
    for fname in os.listdir(dir_path):
        # Skip files that don't match the expected pattern
        if not fname.startswith("results_") or not fname.endswith(".pkl"):
            continue

        # Extract tag from filename (part between 'results_' and '.pkl')
        tag = fname[len("results_"):-len(".pkl")]
        fpath = os.path.join(dir_path, fname)

        # Load pickle file and store in dictionary with tag as key
        with open(fpath, "rb") as f:
            results_dict[tag] = pickle.load(f)

    return results_dict
