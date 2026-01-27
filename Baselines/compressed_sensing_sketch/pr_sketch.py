from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr
import numpy as np

from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime
from Baselines.compressed_sensing_sketch.protype import CompressedSketch


class BloomSketch:
    """A Bloom filter-like sketch implementation using counter vectors."""

    def __init__(self, n_length: int, n_hash: int, key_size: int):
        """
        Initialize a BloomSketch with specified parameters.
        
        Args:
            n_length: Length of the counter vector (will be adjusted to next prime)
            n_hash: Number of hash functions to use
            key_size: Size of keys in bytes
        """
        self.hash_num = n_hash
        self.key_size = key_size
        self.width = calNextPrime(n_length)
        self.h = [GenHashSeed(i) for i in range(n_hash)]
        self.s = [GenHashSeed(i) for i in range(n_hash)]
        self.n = [GenHashSeed(i) for i in range(n_hash)]
        self.counter_vector = np.zeros((self.width,), dtype=int)
    
    def hash(self, key, col):
        """
        Compute hash position for a key and hash function index.
        
        Args:
            key: Key to hash
            col: Hash function index
            
        Returns:
            int: Position in counter vector
        """
        return AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col]) % self.width
    
    def insert(self, key, val=1):
        """
        Insert a key into the sketch or increment its count.
        
        Args:
            key: Key to insert
            val: Value to add to the key's count (default: 1)
            
        Returns:
            int: Always returns 0
            
        Raises:
            AssertionError: If counter would overflow
        """
        for i in range(self.hash_num):
            pos = self.hash(key, i)
            assert self.counter_vector[pos] != np.iinfo(self.counter_vector.dtype).max
            self.counter_vector[pos] += val
        return 0
    
    def get_memory_usage(self):
        """
        Calculate memory usage of the BloomSketch.
        
        Returns:
            int: Memory usage in bytes
        """
        return self.width * self.counter_vector.itemsize
    

class PRSketch(CompressedSketch):
    """Probabilistic Recovery Sketch that extends CompressedSketch with solving capabilities."""

    def __init__(
        self, 
        width: int, 
        depth: int, 
        bf_width: int, 
        bf_hash: int, 
        KEY_T_SIZE: int = 13
    ):
        """
        Initialize a PRSketch with specified dimensions.
        
        Args:
            width: Width of the underlying BloomSketch
            depth: Depth (number of hash functions) of the underlying BloomSketch
            bf_width: Width of the Bloom filter
            bf_hash: Number of hash functions for the Bloom filter
            KEY_T_SIZE: Size of keys in bytes (default: 13)
        """
        sketch = BloomSketch(width, depth, KEY_T_SIZE)
        super().__init__(sketch, KEY_T_SIZE, bf_width, bf_hash)

    def insert(self, key, val=1):
        """
        Insert a key-value pair into the sketch.
        
        Args:
            key: Key to insert
            val: Value to associate with the key (default: 1)
        """
        exist_or_not = self.getbit(key)
        self.sketch.insert(key, val)
        if not exist_or_not:
            self.setbit(key)
            self.flowKeys.append(key)

    def solve_equations(self):
        """
        Solve the compressed sensing equations to recover flow sizes.
        
        Uses LSMR (Least Squares Minimal Residual) algorithm to solve the
        system of linear equations Ax = b, where A is the measurement matrix
        and b is the observed measurements.
        """
        if self.sketchResult != {}:
            return
        
        M = self.sketch.width
        N = len(self.flowKeys)

        A, b = self.return_cs_components(M, N)
        x, i, *_ = lsmr(A, b)
        x[x<0] = 0
        x = np.ceil(np.abs(x)).astype(np.int32)

        for i, key in enumerate(self.flowKeys):
            self.sketchResult[key] = x[i]

    def return_cs_components(self, M: int, N: int):
        """
        Construct the compressed sensing components (measurement matrix and observations).
        
        Args:
            M: Number of measurements (rows in measurement matrix)
            N: Number of signals (columns in measurement matrix)
            
        Returns:
            tuple: (A, b) where A is the measurement matrix and b is the observation vector
        """
        A_data, A_rows, A_cols = [], [], []

        for i in range(self.sketch.hash_num):
            for j, key in enumerate(self.flowKeys):
                idx = self.sketch.hash(key, i)
                A_data.append(1)
                A_rows.append(idx)
                A_cols.append(j)
        
        A = csr_matrix((A_data, (A_rows, A_cols)), shape=(M, N))
        return A, self.sketch.counter_vector

    def query(self, key):
        """
        Query the estimated value for a key.
        
        Args:
            key: Key to query
            
        Returns:
            int: Estimated value for the key
        """
        self.solve_equations()
        exist_or_not = self.getbit(key)

        if exist_or_not:
            try:
                ans = self.sketchResult[key]
            except:
                ans = 1
        else:
            ans = 0

        return ans