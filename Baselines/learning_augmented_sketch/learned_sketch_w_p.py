from Baselines.learning_augmented_sketch.oracle import OracleModel
from Baselines.learning_augmented_sketch.unique_buckets import UniqueBuckets
from Baselines.classic_sketch.count import CountSketch
from Baselines.classic_sketch.count_min import CountMinSketch


class LearnedSketch:
    """
    A hybrid sketch data structure combining learned components with traditional sketches.
    
    This class implements a two-tier sketch system where:
    1. Heavy hitters (frequent items) are tracked in a precise UniqueBuckets structure
    2. Light items (less frequent) are tracked in a CountSketch or CountMinSketch for approximate counting
    
    The combination allows for more accurate frequency estimation across the entire spectrum
    of item frequencies, leveraging the strengths of both precise tracking for heavy hitters
    and space-efficient approximate counting for light items.
    """

    def __init__(self, num_buckets, width, depth, KEY_T_SIZE=13, sketch='cs', **kwargs):
        """
        Initialize the LearnedSketch.
        
        Args:
            num_buckets (int): Number of buckets for the heavy part (UniqueBuckets)
            width (int): Width of the light part (CountSketch or CountMinSketch)
            depth (int): Depth of the light part (CountSketch or CountMinSketch)
            KEY_T_SIZE (int): Size of keys in bytes, defaults to 13
            **kwargs: Additional arguments for OracleModel (currently unused)
        """
        self.set_mode()
        self.classifier = OracleModel(**kwargs)
        if sketch == 'cs':
            self.light_part = CountSketch(width, depth, KEY_T_SIZE)
        elif sketch == 'cm':
            self.light_part = CountMinSketch(width, depth, KEY_T_SIZE)
        else:
            raise ValueError("Invalid sketch type. Choose 'cs' or 'cm'.")
        
        self.heavy_part = UniqueBuckets(num_buckets, KEY_T_SIZE)
        
    def insert(self, key, value=1):
        """
        Insert a key-value pair into the sketch.
        
        The insertion process:
        1. Attempts to insert into the heavy part (UniqueBuckets)
        2. Based on the result, may also insert into the light part (CountSketch or CountMinSketch)
           - If key is new (flag=1): insert into light part
           - If collision causes eviction (flag=2): insert evicted item into light part
        
        Args:
            key: The key to insert
            value (int): The value to add, defaults to 1
        """
        if self.run_oracle:
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
        Query the frequency estimate for a key with an optimization.
        
        This optimized version first checks the heavy part, and only queries
        the light part if the key is not found in the heavy part. This can
        improve performance by avoiding unnecessary computations in the light part.
        
        Args:
            key: The key to query
            
        Returns:
            float: Frequency estimate for the key
        """
        if self.run_oracle:
           device = next(self.classifier.parameters()).device
           _ = self.classifier.run_single_simulation(device)

        # Get precise count from heavy part
        h_ans = self.heavy_part.query(key)
        if h_ans != 0:
            return h_ans
        
        # Get approximate count from light part
        l_ans = self.light_part.query(key)
        return l_ans

    def set_mode(self, run_oracle=False):
        """
        Set the operational mode of the sketch.
        
        Args:
            run_oracle (bool): Whether to run the oracle model, defaults to False
        """
        self.run_oracle = run_oracle

    def get_memory_usage(self):
        """
        Get the memory usage of the sketch.
        
        Returns:
            int: Memory usage in bytes
        """
        if self.run_oracle:
           return self.heavy_part.get_memory_usage() + self.light_part.get_memory_usage() + self.classifier.get_memory_usage()
        
        return self.heavy_part.get_memory_usage() + self.light_part.get_memory_usage()
