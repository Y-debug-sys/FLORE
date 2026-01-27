from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


class Bucket:
    """
    Represents a bucket in the UniqueBuckets data structure.
    
    Each bucket stores a key-value pair along with metadata for collision handling.
    """
    
    def __init__(self, key, value=0, key_size=8):
        """
        Initialize a Bucket instance.
        
        Args:
            key: The key stored in this bucket
            value: The value/count associated with the key
            key_size: Size of the key in bytes
        """
        self.key = key
        self.value = value
        self.key_size = key_size
        self.negative_counter = 0

    @classmethod
    def from_key(cls, key, key_size=8):
        """
        Create a Bucket instance with a specific key.
        
        Args:
            key: The key to store in the bucket
            key_size: Size of the key in bytes
            
        Returns:
            Bucket: New bucket instance with the specified key
        """
        return cls(key, key_size=key_size)

    @classmethod
    def default(cls, key_size=8):
        """
        Create a default (empty) Bucket instance.
        
        Args:
            key_size: Size of the key in bytes
            
        Returns:
            Bucket: New empty bucket instance
        """
        return cls(None, key_size=key_size)

    def get_memory_usage(self):
        """
        Calculate memory usage of this bucket.
        
        Returns:
            int: Memory usage in bytes (2 integers of 4 bytes each + key size)
        """
        return 2 * 4 + self.key_size


class UniqueBuckets:
    """
    A hash table implementation with collision resolution for counting unique items.
    
    Uses a custom hashing scheme with three seeds and handles collisions through
    a negative counter mechanism that evicts less frequent items when needed.
    """

    def __init__(self, num_buckets, KEY_T_SIZE=13):
        """
        Initialize the UniqueBuckets data structure.
        
        Args:
            num_buckets: Number of buckets to allocate (will be rounded up to next prime)
            KEY_T_SIZE: Size of keys in bytes
        """
        self.key_size = KEY_T_SIZE
        self.size = calNextPrime(num_buckets)
        self.buckets = [Bucket.default(KEY_T_SIZE) for _ in range(self.size)]
        self.h, self.s, self.n = [GenHashSeed(i) for i in range(3)]

    def insert(self, temp_key, value=1):
        """
        Insert a key into the hash table or increment its count if it already exists.
        
        Handles three cases:
        1. Key already exists at hashed position: increment value
        2. Position is empty: insert key with value
        3. Collision occurs: use negative counter eviction strategy
        
        Args:
            temp_key: Key to insert or update
            value: Value to add to the key's count (default: 1)
            
        Returns:
            tuple: (status_code, evicted_bucket) where status_code indicates:
                   0 - Key found and updated
                   1 - New key inserted
                   2 - Collision occurred, possibly returning evicted bucket
        """
        temp_bucket = None
        pos = AwareHash(temp_key, self.key_size, self.h, self.s, self.n) % self.size
        
        # Case 1: Key already exists at computed position
        if self.buckets[pos].key == temp_key:
            self.buckets[pos].value += 1
            return 0, temp_bucket
        
        # Case 2: Empty bucket available
        elif self.buckets[pos].key is None:
            self.buckets[pos].value = 1
            self.buckets[pos].key = temp_key
            return 1, temp_bucket
        
        # Case 3: Collision - handle with negative counter eviction strategy
        else:
            temp_bucket = Bucket.default()
            self.buckets[pos].negative_counter += 1
            
            # Evict existing item if negative counter ratio >= 1
            if self.buckets[pos].negative_counter / self.buckets[pos].value >= 1:
                # Copy existing bucket data to temp_bucket before eviction
                temp_bucket.key = self.buckets[pos].key
                temp_bucket.value = self.buckets[pos].value
                temp_bucket.negative_counter = self.buckets[pos].negative_counter
                
                # Replace with new key
                self.buckets[pos] = Bucket(temp_key, value, self.key_size)
                return 2, temp_bucket
            else:
                # Keep existing item, return new key in temp_bucket
                temp_bucket.key = temp_key
                temp_bucket.value = value
                temp_bucket.negative_counter = 0
        
        return 2, temp_bucket
    
    def query(self, temp_key):
        """
        Query the count/value associated with a key.
        
        Args:
            temp_key: Key to look up
            
        Returns:
            int: Value associated with the key, or 0 if key not found
        """
        pos = AwareHash(temp_key, self.key_size, self.h, self.s, self.n) % self.size
        
        if self.buckets[pos].key == temp_key:
            return self.buckets[pos].value
        
        return 0
    
    def get_memory_usage(self):
        """
        Calculate total memory usage of the hash table.
        
        Returns:
            int: Total memory usage in bytes
        """
        return self.size * Bucket.default(key_size=self.key_size).get_memory_usage()
