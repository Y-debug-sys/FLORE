import numpy as np

from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
from sklearn.linear_model import OrthogonalMatchingPursuit

from scipy.sparse import csr_matrix
from Structure.bloom_filter import BloomFilter
from Structure.augmented_filter import HeavyFilter
from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


class FLORESketch:
    """FLORE Sketch implementation for flow reconstruction and frequency estimation.
    
    Combines multiple data structures including Counters, Heavy Filter, and Bloom Filter
    to provide accurate frequency estimates for data streams while maintaining low memory usage.
    """

    def __init__(
        self, 
        depth: int, 
        width: int,
        num_slots: int, 
        num_per_slot: int, 
        bf_width: int, 
        bf_num_hash: int, 
        KEY_T_SIZE: int = 13,
        decoding_method: str = 'flore'
    ):
        """Initialize the FLORESketch with specified parameters.
        
        Args:
            depth: Number of hash functions/rows in the counter matrix
            width: Width of the counter matrix (will be rounded to next prime)
            num_slots: Number of slots in the HeavyFilter
            num_per_slot: Number of entries per slot in the HeavyFilter
            bf_width: Width of the BloomFilter
            bf_num_hash: Number of hash functions for the BloomFilter
            KEY_T_SIZE: Size of the key in bytes (default: 13)
            decoding_method: Method to use for decoding ('flore', 'lsqr', 'lsmr', 'omp', 'local')
        """
        assert decoding_method in ['flore', 'lsqr', 'lsmr', 'omp', 'local'], 'Invalid decoding method'
        self.decoding_method, self.decoding_results = decoding_method, {}

        # Generate hash seeds for each row of counters
        self.h = [GenHashSeed(i) for i in range(depth)]
        self.s = [GenHashSeed(i) for i in range(depth)]
        self.n = [GenHashSeed(i) for i in range(depth)]

        self.key_size, self.tracked_keys = KEY_T_SIZE, []
        self.depth, self.width = depth, calNextPrime(width)

        # Initialize counter matrix with zeros
        self.counters = np.zeros((self.depth, self.width), dtype=int)
        # Initialize HeavyFilter and BloomFilter components
        self.hf = HeavyFilter(num_slots, num_per_slot, KEY_T_SIZE)
        self.bf = BloomFilter(bf_width, bf_num_hash, KEY_T_SIZE)

    def hash(self, key, col):
        """Hash a key to a position in the counter matrix.
        
        Args:
            key: Key to hash
            col: Column (row index) to use for hashing
            
        Returns:
            int: Position in the counter matrix row
        """
        return AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col]) % self.width

    def get_counters(self):
        """Get the counter matrix as float32 array.
        
        Returns:
            numpy.ndarray: Counter matrix converted to float32
        """
        return self.counters.astype(np.float32)

    def get_sketching_matrix(self, to_array=False):
        """Construct the sketching matrix for decoding algorithms.
        
        Returns:
            scipy.sparse.csr_matrix: Sparse matrix representation of the sketch
        """
        Phi_data, Phi_rows, Phi_cols = [], [], []
        M, N = self.get_problem_MN()

        # Populate the sketching matrix with 1s where keys map to counters
        for i in range(self.depth):
            for j, key in enumerate(self.tracked_keys):
                idx = i * self.width + self.hash(key, i)
                Phi_data.append(1)
                Phi_rows.append(idx)
                Phi_cols.append(j)

        Phi = csr_matrix((Phi_data, (Phi_rows, Phi_cols)), shape=(M, N))

        if to_array: return Phi.toarray()
        return Phi

    def insert_counters(self, key, value=1):
        """Insert a key into the counter matrix by incrementing counters.
        
        Args:
            key: Key to insert
            value: Value to add to counters (default: 1)
            
        Raises:
            AssertionError: If any counter would overflow
        """
        for i in range(self.depth):
            pos = self.hash(key, i)
            assert self.counters[i, pos] != np.iinfo(self.counters.dtype).max
            self.counters[i, pos] += value

    def decode(self, key):
        """Decode the frequency estimate for a key using various methods.
        
        Args:
            key: Key to decode frequency for
            
        Returns:
            float: Decoded frequency estimate
        """
        # Return cached result if available
        if self.decoding_results:
            try:
                return self.decoding_results[key]
            except KeyError:
                return 1
            
        if self.decoding_method == 'omp':
            Phi = self.get_sketching_matrix(to_array=True) 
            b = self.get_counters().reshape(-1, )
            omp = OrthogonalMatchingPursuit()
            x = omp.fit(Phi, b).coef_
            x[x < 0] = 0

        elif self.decoding_method == 'lsqr':
            Phi = self.get_sketching_matrix() 
            b = self.get_counters().reshape(-1, )
            x, *_ = lsqr(Phi, b)
            x[x < 0 ] = 0

        else:
            Phi = self.get_sketching_matrix() 
            b = self.get_counters().reshape(-1, )
            x, *_ = lsmr(Phi, b)
            x[x < 0 ] = 0

        # Cache the results for future queries
        for i, key in enumerate(self.tracked_keys):
            self.decoding_results[key] = x[i]
            
        # Try to return the specific key's value
        try:
            return self.decoding_results[key]
        except KeyError:
            return 1

    def insert(self, key, value=1):
        """Insert a key-value pair into the sketch.
        
        Args:
            key: Key to insert
            value: Value to add (default: 1)
        """
        # Insert into HeavyFilter first
        f, v = self.hf.insert(key, value)

        # Process based on HeavyFilter response
        if f == 0:
            # Key was handled entirely by HeavyFilter
            return

        elif f == 1:
            # Key was counted but no swap occurred
            self.insert_counters(key, value)
            # Add to BloomFilter and tracked keys if not already present
            if not self.bf.getbit(key):
                self.bf.setbit(key)
                self.tracked_keys.append(key)

        elif f == -1:
            # A key was evicted from HeavyFilter, handle the evicted key
            swap_key, swap_value = v
            self.insert_counters(swap_key, swap_value)
            # Add the evicted key to BloomFilter and tracked keys if not present
            if not self.bf.getbit(swap_key):
                self.bf.setbit(swap_key)
                self.tracked_keys.append(swap_key)
        
        else:
            raise ValueError('Invalid response from HeavyFilter')

    def query(self, key):
        """Query the frequency estimate for a key.
        
        Args:
            key: Key to query
            
        Returns:
            float: Frequency estimate combining HeavyFilter and decoding results
        """
        # Get result from HeavyFilter
        filter_result = self.hf.query(key)

        # Determine decoding result based on method
        if self.decoding_method == 'local':
            # Local method: minimum of all counter positions
            decoding_result = np.iinfo(self.counters.dtype).max
            for i in range(self.depth):
                pos = self.hash(key, i)
                decoding_result = min(decoding_result, self.counters[i, pos])
        else:
            # Other methods: use BloomFilter to check if key is tracked
            if self.bf.getbit(key):
                decoding_result = self.decode(key)
            else:
                decoding_result = 0
            
        # Return combined result
        return filter_result + decoding_result
    
    def set_decoding_results(self, input_results = None):
        """Set or reset the decoding results cache.
        
        Args:
            input_results: Optional array of results to populate cache with
        """
        if input_results is None: 
            self.decoding_results = {}
        else:
            for i, key in enumerate(self.tracked_keys):
                self.decoding_results[key] = input_results[i]

    def get_memory_usage(self, add_bf=True):
        """Calculate the total memory usage of the sketch.
        
        Args:
            add_bf: Whether to include BloomFilter memory in calculation
            
        Returns:
            float: Total memory usage in bytes
        """
        hf_size = self.hf.get_memory_usage()

        if not add_bf:
            return hf_size + self.depth * self.width * self.counters.itemsize

        bf_size = self.bf.get_memory_usage()
        return hf_size + bf_size + self.depth * self.width * self.counters.itemsize

    def get_problem_MN(self):
        """
        Get the dimensions of the problem for decoding algorithms.
        
        Returns:
            tuple: A tuple containing:
                - int: M dimension, calculated as depth * width
                - int: N dimension, representing the number of tracked keys
        """
        return self.depth * self.width, len(self.tracked_keys)
