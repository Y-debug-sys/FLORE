import numpy as np

from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


class CountSketch:
    """
    Count Sketch implementation for approximate frequency estimation in data streams.
    
    The Count Sketch is a probabilistic data structure that estimates the frequency of elements
    in a data stream. Unlike Count-Min Sketch, it uses pairwise independent hash functions
    and can provide estimates with both overestimation and underestimation, leading to better
    accuracy in practice.
    """

    def __init__(self, width: int, depth: int, KEY_T_SIZE=13):
        """
        Initialize the Count Sketch.
        
        Args:
            width: The width of the sketch matrix (number of columns)
            depth: The depth of the sketch matrix (number of rows/hash functions)
            KEY_T_SIZE: Size of the key in bytes, defaults to 13
        """
        self.key_size = KEY_T_SIZE
        # Ensure width is a prime number for better hash distribution
        self.depth, self.width = depth, calNextPrime(width)
        # Generate hash parameters for positive/negative hash functions
        self.h = [GenHashSeed(i) for i in range(depth)]
        self.s = [GenHashSeed(i) for i in range(depth)]
        self.n = [GenHashSeed(i) for i in range(depth)]
        # Additional hash parameters for sign hash functions
        self.i = [GenHashSeed(i) for i in range(depth)]
        self.j = [GenHashSeed(i) for i in range(depth)]
        self.k = [GenHashSeed(i) for i in range(depth)]
        # Initialize the count matrix with zeros
        self.matrix = np.zeros((self.depth, self.width), dtype=int)

    def hash(self, key, col):
        """
        Hash a key to a position and sign in the specified row of the matrix.
        
        Args:
            key: The key to hash
            col: The column (row index) to use for hashing
            
        Returns:
            tuple: (position, sign) where position is the matrix index and 
                   sign is either +1 or -1
        """
        # First hash function determines the position
        hash_value1 = AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col])
        # Second hash function determines the sign (+1 or -1)
        hash_value2 = AwareHash(key, self.key_size, self.i[col], self.j[col], self.k[col])
        # Return position and sign
        return hash_value1 % self.width, 1 - 2 * (hash_value2 % 2)
    
    def insert(self, key, val=1):
        """
        Insert a key into the sketch or update its count.
        
        Args:
            key: The key to insert/update
            val: The value to add to the key's count, defaults to 1
            
        Raises:
            AssertionError: If the counter would overflow
        """
        for i in range(self.depth):
            pos, sign = self.hash(key, i)
            # Check for potential overflow
            assert self.matrix[i, pos] != np.iinfo(self.matrix.dtype).max
            # Update with signed value
            self.matrix[i, pos] += (val * sign)
    
    def query(self, key):
        """
        Query the estimated frequency of a key.
        
        The estimate is computed by taking the median of signed values from all rows
        to reduce the effect of noise and outliers.
        
        Args:
            key: The key to query
            
        Returns:
            float: Estimated absolute frequency of the key
        """
        # Store results from all hash functions
        results = []
        for i in range(self.depth):
            pos, sign = self.hash(key, i)
            # Retrieve and apply sign to get the estimated value
            results.append(self.matrix[i, pos] * sign)
        # Return the absolute median as the final estimate
        return abs(np.median(results))
    
    def get_memory_usage(self):
        """
        Calculate the memory usage of the sketch.
        
        Returns:
            int: Memory usage in bytes
        """
        return self.depth * self.width * self.matrix.itemsize