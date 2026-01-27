import random

from Baselines.learning_augmented_sketch.oracle import OracleModel
from Baselines.learning_augmented_sketch.unique_buckets import UniqueBuckets
from Baselines.classic_sketch.count import CountSketch


class LearnedSketchW:
    """
    A hybrid sketch data structure with threshold-based filtering.
    
    This variant of the learned sketch applies a threshold to filter out
    low-frequency estimates from the light part (CountSketch), reducing noise
    in the final frequency estimates. It combines:
    1. Precise tracking in heavy part (UniqueBuckets) for frequent items
    2. Approximate tracking in light part (CountSketch) with threshold filtering
    
    The threshold is calculated based on expected noise level in the CountSketch
    to eliminate insignificant estimates.
    """

    def __init__(self, num_buckets, width, depth, KEY_T_SIZE=13, **kwargs):
        """
        Initialize the LearnedSketchW.
        
        Args:
            num_buckets (int): Number of buckets for the heavy part (UniqueBuckets)
            width (int): Width of the light part (CountSketch)
            depth (int): Depth of the light part (CountSketch)
            KEY_T_SIZE (int): Size of keys in bytes, defaults to 13
            **kwargs: Additional arguments for OracleModel (currently unused)
        """
        self.set_mode()
        self.set_nkey(factor=10)
        self.prob, self.classifier = 0.15, OracleModel(**kwargs)
        self.light_part = CountSketch(width, depth, KEY_T_SIZE)
        self.heavy_part = UniqueBuckets(num_buckets, KEY_T_SIZE)
        
    def insert(self, key, value=1):
        """
        Insert a key-value pair into the sketch.
        
        The insertion process:
        1. Attempts to insert into the heavy part (UniqueBuckets)
        2. Based on the result, may also insert into the light part (CountSketch)
           - If key is new (flag=1): insert into light part
           - If collision causes eviction (flag=2): insert evicted item into light part
        
        Args:
            key: The key to insert
            value (int): The value to add, defaults to 1
        """
        if self.run_oracle and random.random() < self.prob:
           device = next(self.classifier.parameters()).device
           _ = self.classifier.run_single_simulation(device)

        # Insert into heavy part and handle the result
        flag, temp_bucket = self.heavy_part.insert(key, value)

        # Flag values:
        # 0 - Key already existed in heavy part, only its count was incremented
        # 1 - New key was inserted into an empty bucket in heavy part
        # 2 - Collision occurred, either evicted existing or returned new key
        if flag == 1:
            # New key inserted into heavy part, also track in light part
            self.light_part.insert(key, value)

        if flag == 2 and temp_bucket is not None:
            # A key was evicted from heavy part, track it in light part
            self.light_part.insert(temp_bucket.key, temp_bucket.value)

    def query(self, key):
        """
        Query the frequency estimate for a key with threshold filtering.
        
        Combines results from both parts of the sketch:
        1. Precise count from heavy part (UniqueBuckets)
        2. Threshold-filtered approximate count from light part (CountSketch)
        
        Low-frequency estimates from the light part are filtered out based
        on a calculated threshold to reduce noise in the final result.
        
        Args:
            key: The key to query
            
        Returns:
            float: Combined frequency estimate (precise count + filtered approximate count)
        """
        if self.run_oracle:
           device = next(self.classifier.parameters()).device
           _ = self.classifier.run_single_simulation(device)

        # Get precise count from heavy part
        h_ans = self.heavy_part.query(key)
        # Get approximate count from light part
        l_ans = self.light_part.query(key)
        
        # Apply threshold filtering to reduce noise from light part
        # Threshold is proportional to expected noise level in CountSketch
        threshold = self.factor * (self.N / (self.light_part.width))
        if l_ans < threshold: l_ans = 0

        # Return combined estimate
        return h_ans + l_ans

    def set_mode(self, run_oracle=False):
        """
        Set the operational mode of the sketch.
        
        Args:
            run_oracle (bool): Whether to run the oracle model, defaults to False
        """
        self.run_oracle = run_oracle

    def set_nkey(self, num_keys=100_000, factor=1.0):
        """
        Set the expected number of keys and threshold factor.
        
        Args:
            num_keys (int): Expected number of distinct keys, defaults to 100,000
            factor (float): Multiplicative factor for threshold calculation, defaults to 1.0
        """
        self.N = num_keys
        self.factor = factor

    def get_memory_usage(self):
        """
        Get the memory usage of the sketch.
        
        Returns:
            int: Memory usage in bytes
        """
        if self.run_oracle:
           return self.heavy_part.get_memory_usage() + self.light_part.get_memory_usage() + self.classifier.get_memory_usage()
        
        return self.heavy_part.get_memory_usage() + self.light_part.get_memory_usage()
    