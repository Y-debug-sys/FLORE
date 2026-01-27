from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.sparse import csr_matrix
import numpy as np

from Baselines.compressed_sensing_sketch.hash_table import HashTable
from Baselines.compressed_sensing_sketch.protype import CompressedSketch
from Baselines.classic_sketch.count_min import CountMinSketch

from sklearn.linear_model import Lasso


class AugmentedSketch:
    """An augmented sketch combining a hash table with a Count-Min sketch for improved accuracy."""

    def __init__(
        self, 
        num_slots: int,
        width: int, 
        depth: int, 
        KEY_T_SIZE: int = 13
    ):
        """
        Initialize an AugmentedSketch with a hash table and Count-Min sketch.
        
        Args:
            num_slots: Number of slots in the hash table
            width: Width of the Count-Min sketch
            depth: Depth of the Count-Min sketch
            KEY_T_SIZE: Size of keys in bytes (default: 13)
        """
        self.hTable = HashTable(num_slots, KEY_T_SIZE)
        self.cm = CountMinSketch(width, depth, KEY_T_SIZE)

    def insert(self, key, val=1):
        """
        Insert a key-value pair into the sketch.
        
        Keys are first inserted into the hash table. If the hash table evicts a key,
        that key is then inserted into the Count-Min sketch.
        
        Args:
            key: Key to insert
            val: Value to associate with the key (default: 1)
            
        Returns:
            tuple: (flag, evicted_key) where flag indicates insertion result and 
                   evicted_key is the key evicted from hash table (if any)
        """
        flag, temp_key = self.hTable.insert(key, val)
        if flag != 0:
            self.cm.insert(temp_key.key, temp_key.val)

        if temp_key is not None:
            return flag, temp_key.key
        else:
            return flag, key

    def get_memory_usage(self):
        """
        Calculate total memory usage of the augmented sketch.
        
        Returns:
            int: Total memory usage in bytes (hash table + Count-Min sketch)
        """
        return self.hTable.get_memory_usage() + self.cm.get_memory_usage()

    
class SeqSketch(CompressedSketch):
    """Sequential sketch implementation using compressed sensing techniques."""

    def __init__(
        self, 
        width: int, 
        depth: int, 
        bf_width: int, 
        bf_hash: int, 
        num_slots: int,
        KEY_T_SIZE: int = 13
    ):
        """
        Initialize a SeqSketch with specified dimensions.
        
        Args:
            width: Width of the underlying Count-Min sketch
            depth: Depth of the underlying Count-Min sketch
            bf_width: Width of the Bloom filter
            bf_hash: Number of hash functions for the Bloom filter
            num_slots: Number of slots in the hash table
            KEY_T_SIZE: Size of keys in bytes (default: 13)
        """
        sketch = AugmentedSketch(num_slots, width, depth, KEY_T_SIZE)
        super().__init__(sketch, KEY_T_SIZE, bf_width, bf_hash)

    def insert(self, key, val=1):
        """
        Insert a key-value pair into the sketch.
        
        Args:
            key: Key to insert
            val: Value to associate with the key (default: 1)
        """
        flag, key = self.sketch.insert(key, val)
        if flag != 0:
            exist_or_not = self.getbit(key)
            if not exist_or_not:
                self.setbit(key)
                self.flowKeys.append(key)

    def solve_equations(self):
        """
        Solve the compressed sensing equations to recover flow sizes.
        
        Uses Orthogonal Matching Pursuit (OMP) algorithm to solve the
        sparse recovery problem Ax = b, where A is the measurement matrix
        and b is the observed measurements.
        """
        if self.sketchResult != {}:
            return
        M = self.sketch.cm.depth * self.sketch.cm.width
        N = len(self.flowKeys)
        A, b = self.return_cs_components(M, N)

        omp = OrthogonalMatchingPursuit()
        print('Start running OMP algorithm...')
        x = omp.fit(A.toarray(), b).coef_
        x[x<0] = 0
        x = np.ceil(np.abs(x)).astype(np.int32)
        print('Done!')

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
        b = np.zeros(M,)
        A_data, A_rows, A_cols = [], [], []

        for i in range(self.sketch.cm.depth):
            for j, key in enumerate(self.flowKeys):
                idx = i * self.sketch.cm.width + self.sketch.cm.hash(key, i)
                A_data.append(1)
                A_rows.append(idx)
                A_cols.append(j)

            for j in range(self.sketch.cm.width):
                b[i * self.sketch.cm.width + j] = self.sketch.cm.matrix[i][j]
        
        A = csr_matrix((A_data, (A_rows, A_cols)), shape=(M, N))
        return A, b

    def query(self, key):
        """
        Query the estimated value for a key.
        
        Combines results from both the hash table (exact counts for keys still in table)
        and the compressed sensing solution (estimated counts for evicted keys).
        
        Args:
            key: Key to query
            
        Returns:
            int: Estimated value for the key
        """
        table_ans = self.sketch.hTable.query(key)
        self.solve_equations()
        exist_or_not = self.getbit(key)

        if exist_or_not:
            try:
                ans = self.sketchResult[key]
            except:
                ans = 1
        else:
            ans = 0

        return table_ans + ans
