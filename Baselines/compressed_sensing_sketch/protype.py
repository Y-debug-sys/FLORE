from bitarray import bitarray
from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


class CompressedSketch:
    """A compressed sensing sketch implementation that combines sketch data structures with Bloom filters."""

    def __init__(
        self,
        sketch,
        key_size: int,
        bf_width: int, 
        bf_hash: int
    ):
        """
        Initialize a CompressedSketch with a base sketch and Bloom filter parameters.
        
        Args:
            sketch: Base sketch data structure to use
            key_size: Size of keys in bytes
            bf_width: Width of the Bloom filter (will be adjusted to next prime)
            bf_hash: Number of hash functions for the Bloom filter
        """
        self.flowKeys = []
        self.sketchResult = {}
        self.sketch = sketch
        self.bf_width = calNextPrime(bf_width)
        self.bit_array = bitarray(self.bf_width)
        self.bf_hash, self.key_size = bf_hash, key_size
        
        self.bit_array.setall(0)
        self.h = [GenHashSeed(i) for i in range(bf_hash)]
        self.s = [GenHashSeed(i) for i in range(bf_hash)]
        self.n = [GenHashSeed(i) for i in range(bf_hash)]

    def insert(self, key, val=1):
        """
        Insert a key-value pair into the sketch.
        
        Args:
            key: Key to insert
            val: Value to associate with the key (default: 1)
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError
        pass

    def return_cs_components(self, M: int, N: int):
        """
        Return compressed sensing components.
        
        Args:
            M: Parameter M for compressed sensing
            N: Parameter N for compressed sensing
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError
        pass
    
    def solve_equations(self):
        """
        Solve equations related to compressed sensing.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError
        pass

    def query(self, key):
        """
        Query the value associated with a key.
        
        Args:
            key: Key to look up
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError
        pass
    
    def get_memory_usage(self, add_bf=True):
        """
        Calculate memory usage of the compressed sketch.
        
        Args:
            add_bf: Whether to include Bloom filter memory in calculation (default: True)
            
        Returns:
            int: Memory usage in bytes
        """
        sketch_size = self.sketch.get_memory_usage()  
        if not add_bf: return sketch_size      
        return sketch_size + (self.bf_width >> 3) + ((self.bf_width & 0x7) != 0)

    def getbit(self, key):
        """
        Check if all bits for a key are set in the Bloom filter.
        
        Args:
            key: Key to check
            
        Returns:
            bool: True if all bits are set, False otherwise
        """
        for i in range(self.bf_hash):
            pos = AwareHash(key, self.key_size, self.h[i], self.s[i], self.n[i]) % self.bf_width
            if not self.bit_array[pos]:
                return False
        return True

    def setbit(self, key):
        """
        Set all bits for a key in the Bloom filter.
        
        Args:
            key: Key for which to set bits
        """
        for i in range(self.bf_hash):
            pos = AwareHash(key, self.key_size, self.h[i], self.s[i], self.n[i]) % self.bf_width
            self.bit_array[pos] = 1

    def clear(self):
        """
        Clear the sketch result cache.
        
        Resets the sketchResult dictionary which stores the results of 
        compressed sensing equation solving. This method is typically 
        called to free memory or reset the state between experiments.
        """
        self.sketchResult = {}
