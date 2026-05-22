import math
import numpy as np

# from Baselines.learning_augmented_sketch.oracle import OracleModel


# Constants for sketch configurations
NUM_SKETCH_HASH_FUNCTIONS = 4      # Default number of hash functions for sketches
NUM_BLOOM_HASH_FUNCTIONS = 7       # Default number of hash functions for Bloom filters
NUM_ENTRY = 64                     # Default number of entries per slot
ITEM_SIZE = np.zeros((1,), dtype=int).itemsize  # Size of an integer item in bytes

# Memory budget configurations in bytes
BUDGETS = {
    '16KB'  : 16 << 10,    # 16 KB
    '32KB'  : 32 << 10,    # 32 KB
    '48KB'  : 48 << 10,    # 48 KB
    '64KB'  : 64 << 10,    # 64 KB
    '80KB'  : 80 << 10,    # 80 KB
    '96KB'  : 96 << 10,    # 96 KB
    '112KB' : 112 << 10,   # 112 KB
    '128KB' : 128 << 10,   # 128 KB
    '256KB' : 256 << 10,   # 256 KB
    '512KB' : 512 << 10,   # 512 KB
    '1MB'   : 1024 << 10,  # 1 MB
    '2MB'   : 2048 << 10,  # 2 MB
}


def get_bloom_filter_bits(num_items, alpha):
    """
    Calculate optimal number of bits for a Bloom filter.
    
    Uses the formula for optimal Bloom filter size:
    m = -(k * n) / ln(1 - α^(1/k))
    where k is the number of hash functions, n is the number of items,
    and α is the desired false positive rate.
    
    Args:
        num_items (int): Expected number of items to be stored
        alpha (float): Desired false positive rate (between 0 and 1)
        
    Returns:
        int: Optimal number of bytes for the Bloom filter
    """
    m_opt = - (NUM_BLOOM_HASH_FUNCTIONS * num_items) / math.log(1 - alpha**(1 / NUM_BLOOM_HASH_FUNCTIONS))
    return int(math.ceil(m_opt) >> 3)  # Convert bits to bytes with ceiling division


def cm_or_cs_config(memory_bytes): 
    """
    Configure parameters for Count-Min or Count-Sketch data structures.
    
    This function calculates the optimal configuration for Count-Min or Count-Sketch
    based on the given memory budget. The data structure is composed of a 2D array
    where depth corresponds to the number of hash functions and width corresponds
    to the number of counters per hash function.
    
    Memory allocation formula:
    - Total memory = depth * width * ITEM_SIZE
    - width = memory_bytes // (depth * ITEM_SIZE)
    - depth is fixed by NUM_SKETCH_HASH_FUNCTIONS constant
    
    Args:
        memory_bytes (int): Total memory available in bytes
        
    Returns:
        dict: Configuration dictionary containing:
            - depth (int): Number of hash functions (rows in the sketch)
            - width (int): Number of counters per hash function (columns in the sketch)
            
    Example:
        >>> cm_or_cs_config(1024)
        {'depth': 4, 'width': 256}
        
    Note:
        The actual width may be smaller than theoretical maximum due to integer
        division truncation when dividing available memory by item size.
    """
    width = memory_bytes // (NUM_SKETCH_HASH_FUNCTIONS * ITEM_SIZE) 
    return {"depth": NUM_SKETCH_HASH_FUNCTIONS, "width": width} 


# def ls_config(memory_bytes, key_size=4, heavy_ratio=0.5, has_model=False): 
#     """
#     Configure parameters for Learned Sketch data structure.
    
#     Divides memory between heavy part (H) and light part (L) based on heavy_ratio.
#     The heavy part stores key-value pairs, while the light part uses a sketch structure.
    
#     Args:
#         memory_bytes (int): Total memory available in bytes
#         key_size (int): Size of each key in bytes (default: 4)
#         heavy_ratio (float): Ratio of memory allocated to heavy part (default: 0.5)
#         has_model (bool): Whether to account for model memory usage (default: False)
        
#     Returns:
#         dict: Configuration dictionary with num_buckets, depth, and width parameters
#     """
#     # Deduct model memory if present
#     if has_model: 
#         model_size = OracleModel().get_memory_usage()
#         memory_bytes = memory_bytes - model_size

#     # Split memory between heavy (H) and light (L) parts
#     mem_H = int(memory_bytes * heavy_ratio)  # Calculate heavy part memory using multiplication
#     mem_L = memory_bytes - mem_H 

#     # Calculate number of buckets for heavy part (each bucket stores key + 2 4-byte values)
#     num_buckets = mem_H // (2 * 4 + key_size)
#     # Calculate width for light sketch part
#     width = mem_L // (NUM_SKETCH_HASH_FUNCTIONS * ITEM_SIZE)
#     return {"num_buckets": num_buckets, "depth": NUM_SKETCH_HASH_FUNCTIONS, "width": width}


def es_config(memory_bytes, key_size=4, heavy_ratio=0.5):
    """
    Configure parameters for Elastic Sketch data structure.
    
    This function allocates memory between the heavy part (buckets with entries) 
    and light part (Count-Min Sketch) of the Elastic Sketch based on the specified 
    heavy_ratio. The Elastic Sketch uses a bucket-based heavy hitter detection 
    mechanism combined with a Count-Min sketch for light items.
    
    Memory allocation breakdown:
    - Heavy part: Stores buckets, each containing NUM_ENTRY entries
    - Each entry requires (key_size + 4) bytes (key + 4-byte integer value)
    - Light part: Uses Count-Min sketch with NUM_SKETCH_HASH_FUNCTIONS depth
    
    Args:
        memory_bytes (int): Total memory available in bytes for the entire Elastic Sketch
        key_size (int, optional): Size of each flow key in bytes. Defaults to 4.
        heavy_ratio (float, optional): Ratio of total memory allocated to the heavy part. 
                                     Defaults to 0.5 (50% to heavy, 50% to light).
        
    Returns:
        dict: Configuration dictionary containing:
            - num_buckets (int): Number of buckets in the heavy part
            - num_per_bucket (int): Number of entries per bucket (fixed to NUM_ENTRY constant)
            - width (int): Width of the Count-Min sketch (light part)
            - depth (int): Depth of the Count-Min sketch (fixed to NUM_SKETCH_HASH_FUNCTIONS)
            
    Memory calculation details:
        mem_H = int(memory_bytes * heavy_ratio)  # Heavy part memory allocation
        mem_L = memory_bytes - mem_H             # Light part memory allocation
        num_buckets = mem_H // (NUM_ENTRY * (key_size + 4))  # Buckets that fit in heavy memory
        width = mem_L // (NUM_SKETCH_HASH_FUNCTIONS * ITEM_SIZE)  # Sketch width from light memory
        
    Note:
        - The actual memory usage may be slightly less than memory_bytes due to integer division
        - NUM_ENTRY is a global constant (default: 64) defining entries per bucket
        - This configuration matches the ElasticSketch class constructor parameters
    """
    mem_H = int(memory_bytes * heavy_ratio)  # Calculate heavy part memory using multiplication
    mem_L = memory_bytes - mem_H 
    width = mem_L // (NUM_SKETCH_HASH_FUNCTIONS * ITEM_SIZE)

    # Allocate remaining memory to heavy part slots
    num_buckets = mem_H // (NUM_ENTRY * (key_size + 4))
    return {"num_buckets": num_buckets, "num_per_bucket": NUM_ENTRY, "width": width, "depth": NUM_SKETCH_HASH_FUNCTIONS}


def ag_config(memory_bytes, key_size=4, heavy_ratio=0.5, max_len=100): 
    """
    Configure parameters for Augmented Sketch data structure.
    
    This function divides the total memory budget between a heavy part (H) and a light part (L).
    The heavy part stores key-value pairs directly, while the light part uses a sketch structure
    for approximate counting. It calculates the optimal configuration based on memory constraints.
    
    Args:
        memory_bytes (int): Total memory available in bytes
        key_size (int, optional): Size of each key in bytes. Defaults to 4.
        heavy_ratio (float, optional): Ratio of memory allocated to heavy part. Defaults to 0.5.
        max_len (int, optional): Maximum length of the heavy part. Defaults to 100.
        
    Returns:
        dict: Configuration dictionary containing:
            - num_entries (int): Number of entries in the heavy part
            - depth (int): Number of hash functions for the light sketch part
            - width (int): Width of the light sketch part
    """
    mem_H = int(memory_bytes * heavy_ratio)

    num_entries = mem_H // (2 * 4 + key_size)
    num_entries = max_len if num_entries > max_len else num_entries

    mem_L = memory_bytes - num_entries * (2 * 4 + key_size)
    width = mem_L // (NUM_SKETCH_HASH_FUNCTIONS * ITEM_SIZE)
    return {"num_entries": num_entries, "depth": NUM_SKETCH_HASH_FUNCTIONS, "width": width}


def pr_config(
    memory_bytes: int,
    stream_size: int = 100_000,
    has_bloom: bool = True
): 
    """
    Configure parameters for PR (Probably learned Relaxed) Sketch data structure.
    
    Allocates memory between a Bloom filter and sketch structure, with the Bloom filter
    helping to reduce false positives in the sketch. The memory allocation strategy
    depends on whether a Bloom filter is used and splits memory approximately evenly
    between the Bloom filter and the sketch when applicable.
    
    Args:
        memory_bytes (int): Total memory available in bytes for the sketch structure
        stream_size (int): Expected number of items in the stream (default: 100,000)
        has_bloom (bool): Whether to include a Bloom filter in the configuration (default: True)
        
    Returns:
        dict: Configuration dictionary containing:
            - bf_width: Width of the Bloom filter in bits
            - width: Width of the sketch structure
            - bf_hash: Number of hash functions for the Bloom filter
    """
    if has_bloom: 
        bf_mem_upper = get_bloom_filter_bits(stream_size, 0.05)
        mem_bf = min(bf_mem_upper, memory_bytes // 2)
        memory_bytes = memory_bytes - mem_bf
    else:
        mem_bf = get_bloom_filter_bits(stream_size, 0.001)

    bf_width = int(math.ceil(mem_bf) * 8)  # Convert bytes to bits
    width = memory_bytes // ITEM_SIZE
    return {"bf_width": bf_width, "width": width, "depth": NUM_SKETCH_HASH_FUNCTIONS, "bf_hash": NUM_BLOOM_HASH_FUNCTIONS}


def nze_config(
    memory_bytes: int, 
    key_size: int = 4, 
    stream_size: int = 100_000,
    has_bloom: bool = True,
    light_upper: float = 64 << 10
    # light_upper: float = 1024 << 10
):
    """
    Configure parameters for NZE (Non-Zero Estimator) Sketch data structure.
    
    Divides memory allocation among three components:
    1. Bloom filter for approximate membership testing
    2. Light sketch part for frequency estimation
    3. Slots for storing actual key-value pairs
    
    Memory is distributed with constraints to ensure efficient utilization
    of all components while respecting upper bounds on certain parts.
    
    Args:
        memory_bytes (int): Total memory available in bytes
        key_size (int): Size of each key in bytes (default: 4)
        stream_size (int): Expected number of items in the stream (default: 100,000)
        has_bloom (bool): Whether to include a Bloom filter (default: True)
        light_upper (float): Upper limit for light part memory (default: 64KB)
        
    Returns:
        dict: Configuration dictionary containing:
            - num_slots: Number of slots for storing key-value pairs
            - bf_width: Width of the Bloom filter in bits
            - width: Width of the light sketch structure
            - bf_hash: Number of hash functions for the Bloom filter
            - depth: Depth (number of hash functions) of the light sketch
    """
    if has_bloom: 
        bf_mem_upper = get_bloom_filter_bits(stream_size, 0.05)
        mem_bf = min(bf_mem_upper, memory_bytes // 2)
        memory_bytes = memory_bytes - mem_bf
    else:
        mem_bf = get_bloom_filter_bits(stream_size, 0.001)

    bf_width = int(math.ceil(mem_bf) * 8)  # Convert bytes to bits
    mem_light = min(light_upper, memory_bytes // 2)
    width = mem_light // (NUM_SKETCH_HASH_FUNCTIONS * ITEM_SIZE)

    memory_bytes = memory_bytes - mem_light
    num_slots = memory_bytes // (2 * 4 + key_size)
    
    return {"num_slots": num_slots, "bf_width": bf_width, "width": width,
            "bf_hash": NUM_BLOOM_HASH_FUNCTIONS, "depth": NUM_SKETCH_HASH_FUNCTIONS}


def flore_config(
    memory_bytes: int,
    key_size: int = 4, 
    light_upper: float = 64 << 10, 
    stream_size: int = 100_000, 
    has_bloom: bool = True
): 
    """
    Configure parameters for FLORE data structure.
    
    FLORE consists of multiple components:
    - Bloom filter for membership testing
    - Light sketch for frequency estimation
    - Heavy part for storing frequent items
    
    Args:
        memory_bytes (int): Total memory available in bytes
        light_upper (float): Upper limit for light part memory (default: 64KB)
        stream_size (int): Expected stream size (default: 100,000)
        has_bloom (bool): Whether to include a Bloom filter (default: True)
        
    Returns:
        dict: Configuration dictionary with all FLORE parameters
    """
    # Configure Bloom filter memory
    if has_bloom: 
        bf_mem_upper = get_bloom_filter_bits(stream_size, 0.05)
        mem_bf = min(bf_mem_upper, memory_bytes // 2)
        memory_bytes = memory_bytes - mem_bf
    else:
        # Even if not using Bloom filter, reserve memory for it
        mem_bf = get_bloom_filter_bits(stream_size, 0.001)

    # Calculate Bloom filter width in bits
    bf_width = int(math.ceil(mem_bf) << 3)
    
    # Configure light sketch memory (with upper bound)
    mem_light = min(light_upper, memory_bytes // 2)
    width = mem_light // (NUM_SKETCH_HASH_FUNCTIONS * ITEM_SIZE)

    # Allocate remaining memory to heavy part slots
    memory_bytes = memory_bytes - mem_light
    num_slots = memory_bytes // (NUM_ENTRY * (key_size + 4))
    
    return {"num_slots": num_slots, "num_per_slot": NUM_ENTRY, "bf_width": bf_width, "width": width,
            "bf_num_hash": NUM_BLOOM_HASH_FUNCTIONS, "depth": NUM_SKETCH_HASH_FUNCTIONS}


if __name__ == '__main__':
    pass
