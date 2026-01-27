import os
import numpy as np


def syn_zipf(n, a=1, scale=200):
    """
    Generate synthetic data following Zipf distribution.
    
    The Zipf distribution follows the probability law where the frequency of an item is 
    inversely proportional to its rank in a frequency table. This function creates
    synthetic data where item frequencies follow this pattern.
    
    Parameters:
    - n (int): Number of distinct items to generate
    - a (float): Exponent parameter of the Zipf distribution (default: 1)
    - scale (int): Scaling factor for frequency values (default: 1000)
    
    Returns:
    - list: List of frequencies following Zipf distribution, randomly shuffled
    """
    frequencies = np.array([int(n/(i**a)*scale) for i in range(1, n+1)])
    shuffled_indices = np.random.permutation(len(frequencies))
    return frequencies[shuffled_indices].tolist()


def syn_pareto(n, alpha=1.5, scale=1000):
    """
    Generate synthetic data following Pareto distribution.
    
    The Pareto distribution is a power-law probability distribution that is used
    to describe social, scientific, geophysical, actuarial, and many other types
    of observable phenomena. It is sometimes referred to as the "80-20 rule".
    
    Parameters:
    - n (int): Number of samples to generate
    - alpha (float): Shape parameter of the Pareto distribution (default: 1.5)
    - scale (int): Scaling factor for the distribution (default: 2000)
    
    Returns:
    - list: List of samples following Pareto distribution, randomly shuffled
    """
    samples = scale * (1 + np.random.pareto(alpha, size=n))
    freq = samples.astype(int)
    s = np.random.permutation(n)
    return freq[s].tolist()


def syn_lognormal(n, mean=0.0, sigma=1.0, scale=2000):
    """
    Generate synthetic data following Log-normal distribution.
    
    A log-normal distribution is a continuous probability distribution of a random variable
    whose logarithm is normally distributed. It is commonly used to model growth processes
    and financial data among others.
    
    Parameters:
    - n (int): Number of samples to generate
    - mean (float): Mean of the underlying normal distribution (default: 0.0)
    - sigma (float): Standard deviation of the underlying normal distribution (default: 1.0)
    - scale (int): Scaling factor for the distribution (default: 2000)
    
    Returns:
    - list: List of samples following Log-normal distribution, randomly shuffled
    """
    samples = scale * (1 + np.random.lognormal(mean, sigma, size=n))
    freq = samples.astype(int)
    s = np.random.permutation(n)
    return freq[s].tolist()


def syn_exponential(n, lam=0.25, scale=1000):
    """
    Generate synthetic data following Exponential distribution.
    
    The exponential distribution is the probability distribution of the time between events
    in a Poisson point process, i.e., a process in which events occur continuously and 
    independently at a constant average rate.
    
    Parameters:
    - n (int): Number of samples to generate
    - lam (float): Rate parameter (lambda) of the exponential distribution (default: 0.25)
    - scale (int): Scaling factor for the distribution (default: 2000)
    
    Returns:
    - list: List of samples following Exponential distribution, randomly shuffled
    """
    samples = scale * (1 + np.random.exponential(1/lam, size=n))
    freq = samples.astype(int)
    s = np.random.permutation(n)
    return freq[s].tolist()


def read_traces(path: str, key_size: int = 13):
    """
    Read 5-tuple network traces from a .dat file
    
    This function reads binary data representing network packet traces from a file.
    Each trace entry is expected to be of fixed size.
    
    Args:
        path (str): Path to the file containing traces
        key_size (int): Size of each trace in bytes
    
    Returns:
        tuple: A tuple containing the number of traces and a list of traces
    """
    assert os.path.isfile(path), "File does not exist"
    traces = []

    with open(path, 'rb') as input_data:
        while True:
            str_data = input_data.read(key_size)
            if len(str_data) < key_size:
                break
            traces.append(str_data)
    
    return len(traces), traces


def read_items(path: str, key_size: int = 4):
    """
    Read streaming data from a .dat file
    
    This function reads text data from a file where each line contains space-separated integers.
    These integers are converted to byte representations and concatenated into a single list.
    
    Args:
        path (str): Path to the file containing items
        key_size (int): Size of each item in bytes
    
    Returns:
        tuple: A tuple containing the number of items and a list of items
    """
    items = []
    with open(path, 'r') as file:
        data = file.readlines()

    for line in data:
        # Parse space-separated integers from each line
        integers = list(map(int, line.strip().split()))
        # Convert each integer to bytes and add to items list
        items += [i.to_bytes(key_size, 'little') for i in integers]
    
    return len(items), items


def gen_stream(num_items: int, key_size: int = 4, seed: int = 42, type: str = 'zipf'):
    """
    Generate streaming data based on various statistical distributions
    
    This function creates synthetic streaming data using different probability 
    distributions. The generated data represents item frequencies that follow 
    the specified distribution pattern.
    
    Args:
        num_items (int): Number of items to generate
        key_size (int): Size of each item in bytes
        seed (int): Seed for random number generator to ensure reproducibility
        type (str): Type of distribution to use for generating items. 
                   Must be one of: 'zipf', 'pareto', 'lognormal', 'exponential'
    
    Returns:
        tuple: A tuple containing the number of items and a list of items represented as bytes
    """
    assert type in ['zipf', 'pareto', 'lognormal', 'exponential', 'zipf-icml'], "Invalid type"
    # Set seed for reproducible results
    np.random.seed(seed)
    items = []

    # Select distribution based on type parameter
    if type == 'zipf-icml':
        data = syn_zipf(num_items)
    elif type == 'pareto':
        data = syn_pareto(num_items)
    elif type == 'lognormal':
        data = syn_lognormal(num_items)
    elif type == 'exponential':
        data = syn_exponential(num_items)
    elif type == 'zipf':
        data = np.random.zipf(1.4, size=num_items).tolist()
    else:
        raise ValueError("Invalid distribution type")

    # Convert integer data to byte representation
    items += [i.to_bytes(key_size, 'little') for i in data]
    return len(items), items


def restore_int_from_bytes(byte_data: bytes):
    """
    Convert byte representation of an integer to its original integer value
    
    This function takes a byte representation of an integer and converts it back to its original integer value.
    
    Args:
        byte_data (bytes): Byte representation of an integer
    
    Returns:
        int: Original integer value
    """
    return int.from_bytes(byte_data, 'little')


if __name__ == "__main__":
    length, _ = gen_stream(num_items=100000, type='exponential')
    print(length)