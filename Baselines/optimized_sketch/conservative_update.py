import numpy as np

from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


class ConservativeUpdateSketch:
    """
    A sketch implementation using conservative update strategy to reduce overestimation.
    
    The conservative update technique updates only the counter that attains the minimum value 
    among all the counters indexed by the hash functions, and ensures that all counters stay 
    at the same level to minimize error. This approach significantly reduces the impact of 
    collisions and overestimation compared to standard sketches.
    
    Attributes:
        key_size (int): Size of the key in bytes
        depth (int): Number of hash functions
        width (int): Size of each row in the sketch matrix
        h (list): Hash seeds for primary hashing
        s (list): Hash seeds for secondary hashing
        n (list): Hash seeds for tertiary hashing
        matrix (np.ndarray): 2D array representing the sketch counters
    """

    def __init__(self, width: int, depth: int, KEY_T_SIZE=13):
        """
        Initialize the Conservative Update Sketch.
        
        Args:
            width (int): Width of the sketch matrix (will be rounded up to next prime)
            depth (int): Depth of the sketch matrix (number of hash functions)
            KEY_T_SIZE (int, optional): Key size in bytes. Defaults to 13.
        """
        self.key_size = KEY_T_SIZE
        self.depth, self.width = depth, calNextPrime(width)
        self.h = [GenHashSeed(i) for i in range(depth)]
        self.s = [GenHashSeed(i) for i in range(depth)]
        self.n = [GenHashSeed(i) for i in range(depth)]
        self.matrix = np.zeros((self.depth, self.width), dtype=int)

    def hash(self, key, col):
        """
        Hash a key using the specified hash function index.
        
        Args:
            key (bytes): Key to be hashed
            col (int): Index of the hash function to use
            
        Returns:
            int: Position in the sketch matrix after hashing
        """
        return AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col]) % self.width
    
    def insert(self, key, val=1):
        """
        Insert a key into the sketch with conservative update strategy.
        
        The algorithm finds the position with minimum counter value and increments it,
        then ensures all positions store at least that value to reduce overestimation.
        
        Args:
            key (bytes): Key to be inserted
            val (int, optional): Value to increment by. Defaults to 1.
        """
        positions, hits = [], []
        for i in range(self.depth):
            pos = self.hash(key, i)
            positions.append(pos)
            hits.append(self.matrix[i, pos])

        index = np.argmin(hits)
        self.matrix[index, positions[index]] += val
        min_value = self.matrix[index, positions[index]]

        for i in range(self.depth):
            self.matrix[i, positions[i]] = max(min_value, self.matrix[i, positions[i]])
    
    def query(self, key):
        """
        Query the frequency estimation of a key from the sketch.
        
        Returns the minimum value among all positions indexed by the key's hash functions,
        which provides an upper bound estimate of the actual frequency.
        
        Args:
            key (bytes): Key to query
            
        Returns:
            int: Estimated frequency of the key
        """
        result = np.iinfo(self.matrix.dtype).max
        for i in range(self.depth):
            pos = self.hash(key, i)
            result = min(result, self.matrix[i, pos])
        return result
    
    def get_memory_usage(self):
        """
        Calculate the memory usage of the sketch.
        
        Returns:
            int: Memory usage in bytes
        """
        return self.depth * self.width * self.matrix.itemsize