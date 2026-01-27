import numpy as np

from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


class CountMinSketch:
    """
    Count-Min Sketch implementation for approximate frequency counting in data streams.
    
    The Count-Min Sketch is a probabilistic data structure that serves as a frequency table
    to estimate the frequency of elements in a data stream using sub-linear space.
    """

    def __init__(self, width: int, depth: int, KEY_T_SIZE=13):
        """
        Initialize the Count-Min Sketch.
        
        Args:
            width: The width of the sketch matrix (number of columns)
            depth: The depth of the sketch matrix (number of rows/hash functions)
            KEY_T_SIZE: Size of the key in bytes, defaults to 13
        """
        self.key_size = KEY_T_SIZE
        # Ensure width is a prime number for better hash distribution
        self.depth, self.width = depth, calNextPrime(width)
        # Generate hash parameters for each row
        self.h = [GenHashSeed(i) for i in range(depth)]
        self.s = [GenHashSeed(i) for i in range(depth)]
        self.n = [GenHashSeed(i) for i in range(depth)]
        # Initialize the count matrix with zeros
        self.matrix = np.zeros((self.depth, self.width), dtype=int)

    def hash(self, key, col):
        """
        Hash a key to a position in the specified row of the matrix.
        
        Args:
            key: The key to hash
            col: The column (row index) to use for hashing
            
        Returns:
            int: The hashed position in the matrix
        """
        return AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col]) % self.width
    
    def insert(self, key, val=1):
        """
        Insert a key into the sketch or increment its count.
        
        Args:
            key: The key to insert/update
            val: The value to add to the key's count, defaults to 1
            
        Raises:
            AssertionError: If the counter would overflow
        """
        for i in range(self.depth):
            pos = self.hash(key, i)
            # Check for potential overflow
            assert self.matrix[i, pos] != np.iinfo(self.matrix.dtype).max
            self.matrix[i, pos] += val
    
    def query(self, key):
        """
        Query the estimated frequency of a key.
        
        The estimate is the minimum value across all rows to reduce overestimation
        due to hash collisions.
        
        Args:
            key: The key to query
            
        Returns:
            int: Estimated frequency of the key
        """
        # Initialize with maximum possible value
        result = np.iinfo(self.matrix.dtype).max
        for i in range(self.depth):
            pos = self.hash(key, i)
            # Take minimum across all hash functions to reduce error
            result = min(result, self.matrix[i, pos])
        return result
    
    def get_memory_usage(self):
        """
        Calculate the memory usage of the sketch.
        
        Returns:
            int: Memory usage in bytes
        """
        return self.depth * self.width * self.matrix.itemsize
    