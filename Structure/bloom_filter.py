from bitarray import bitarray
from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


class BloomFilter:
    """
    Bloom Filter implementation for probabilistic set membership testing.
    
    A Bloom filter is a space-efficient probabilistic data structure that is used to test
    whether an element is a member of a set. False positive matches are possible, but
    false negatives are not. In other words, a query returns either "possibly in set" or
    "definitely not in set".
    """
    
    def __init__(self, w, hash_num, KEY_T_SIZE=8):
        """
        Initialize the Bloom Filter.
        
        Args:
            w: Width parameter for the Bloom filter
            hash_num: Number of hash functions to use
            KEY_T_SIZE: Size of the key in bytes, defaults to 8
        """
        self.key_size = KEY_T_SIZE
        # Ensure width is a prime number for better hash distribution
        self.width = calNextPrime(w)
        # Calculate the size in bytes needed to store width bits
        self.size = (self.width >> 3) + ((self.width & 0x7) != 0)
        # Initialize the bit array with zeros
        self.bit_array = bitarray(self.size * 8)
        self.bit_array.setall(0)
        # Generate hash parameters for each hash function
        self.h = [GenHashSeed(i) for i in range(hash_num)]
        self.s = [GenHashSeed(i) for i in range(hash_num)]
        self.n = [GenHashSeed(i) for i in range(hash_num)]
        self.hash_num = hash_num
    
    def getbit(self, k):
        """
        Check if a key is possibly in the set (may return false positives).
        
        Args:
            k: Key to check for membership
            
        Returns:
            bool: True if key may be in the set, False if definitely not in the set
        """
        # Check all hash positions - if any is 0, key is definitely not in set
        for i in range(self.hash_num):
            pos = AwareHash(k, self.key_size, self.h[i], self.s[i], self.n[i]) % self.width
            if not self.bit_array[pos]:
                return False
        # All positions are set, key may be in the set
        return True
    
    def setbit(self, k):
        """
        Add a key to the set.
        
        Args:
            k: Key to add to the set
        """
        # Set all hash positions to 1
        for i in range(self.hash_num):
            pos = AwareHash(k, self.key_size, self.h[i], self.s[i], self.n[i]) % self.width
            self.bit_array[pos] = 1
    
    def reset(self):
        """
        Reset the Bloom filter by clearing all bits.
        """
        self.bit_array.setall(0)
    
    def get_memory_usage(self):
        """
        Get the memory usage of the Bloom filter in bytes.
        
        Returns:
            int: Memory usage in bytes
        """
        return self.size
    
    def get_hash_num(self):
        """
        Get the number of hash functions used.
        
        Returns:
            int: Number of hash functions
        """
        return self.hash_num