from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


# Constants for insertion results
# HIT: Key found and incremented
# MISS_EVICT: Key not found, evicted existing entry
# MISS_INSERT: Key not found, could not insert due to threshold
HIT, MISS_EVICT, MISS_INSERT = 0, 1, 2
EVICT_THRESHOLD = 1


class slot:
    """Represents a hash table slot containing a key-value pair."""
    
    def __init__(self, key, val=0, key_size=8):
        """
        Initialize a slot with a key, value, and key size.
        
        Args:
            key: The key to store in the slot
            val: Initial value/count for the key (default: 0)
            key_size: Size of the key in bytes (default: 8)
        """
        self.key = key
        self.val = val
        self.key_size = key_size
        self.negative_counter = 0

    @classmethod
    def from_key(cls, key, key_size=8):
        """
        Create a slot instance from a key with default values.
        
        Args:
            key: The key to store in the slot
            key_size: Size of the key in bytes (default: 8)
            
        Returns:
            slot: New slot instance with the given key
        """
        return cls(key, key_size=key_size)

    @classmethod
    def default(cls, key_size=8):
        """
        Create a default empty slot.
        
        Args:
            key_size: Size of the key in bytes (default: 8)
            
        Returns:
            slot: New empty slot instance
        """
        return cls(None, key_size=key_size)

    def get_memory_usage(self):
        """
        Calculate memory usage of the slot.
        
        Returns:
            int: Memory usage in bytes (2 integers of 4 bytes each plus key size)
        """
        return 2 * 4 + self.key_size


class HashTable:
    """A hash table implementation with eviction capabilities based on negative counters."""

    def __init__(self, slot_num, KEY_T_SIZE=13):
        """
        Initialize a hash table with a given number of slots.
        
        Args:
            slot_num: Initial number of slots in the hash table
            KEY_T_SIZE: Size of keys in bytes (default: 13)
        """
        self.key_size = KEY_T_SIZE
        self.size = calNextPrime(slot_num)
        self.slots = [slot.default(key_size=KEY_T_SIZE) for _ in range(self.size)]
        self.h, self.s, self.n = [GenHashSeed(i) for i in range(3)]

    def insert(self, temp_key, val=1):
        """
        Insert a key into the hash table or increment its count if it exists.
        
        Args:
            temp_key: Key to insert or update
            val: Value to add to the key's count (default: 1)
            
        Returns:
            tuple: (status, evicted_slot) where status is HIT, MISS_EVICT, or MISS_INSERT,
                   and evicted_slot contains evicted data if applicable
        """
        temp_slot = None
        pos = AwareHash(temp_key, self.key_size, self.h, self.s, self.n) % self.size
        
        if self.slots[pos].key == temp_key:
            # Key found, increment value
            self.slots[pos].val += val
            return HIT, temp_slot
        elif self.slots[pos].key is None:
            # Empty slot, insert new key
            self.slots[pos].val = val
            self.slots[pos].key = temp_key
            return HIT, temp_slot
        else:
            # Collision occurred, handle eviction logic
            temp_slot = slot.default()
            self.slots[pos].negative_counter += val
            if self.slots[pos].negative_counter / self.slots[pos].val >= EVICT_THRESHOLD:
                # Evict existing entry and insert new one
                temp_slot.key = self.slots[pos].key
                temp_slot.val = self.slots[pos].val
                temp_slot.negative_counter = self.slots[pos].negative_counter
                self.slots[pos] = slot(temp_key, val, self.key_size)
                return MISS_EVICT, temp_slot
            else:
                # Don't evict, return new key info
                temp_slot.key = temp_key
                temp_slot.val = val
                temp_slot.negative_counter = 0
        
        return MISS_INSERT, temp_slot
    
    def query(self, temp_key):
        """
        Query the count/value for a given key.
        
        Args:
            temp_key: Key to look up
            
        Returns:
            int: Value/count associated with the key, or 0 if not found
        """
        pos = AwareHash(temp_key, self.key_size, self.h, self.s, self.n) % self.size
        
        if self.slots[pos].key == temp_key:
            return self.slots[pos].val
        
        return 0
    
    def get_memory_usage(self):
        """
        Calculate total memory usage of the hash table.
        
        Returns:
            int: Total memory usage in bytes
        """
        return self.size * slot.default(key_size=self.key_size).get_memory_usage()
