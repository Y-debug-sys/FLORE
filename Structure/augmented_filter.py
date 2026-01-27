from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


class Entry:
    """Represents an entry in the HeavyFilter containing a key-value pair.
    
    Attributes:
        key: The key stored in this entry, or None if empty
        value: The value/count associated with the key
        # flag: A boolean flag indicating special status (e.g., swapped out)
    """
    def __init__(self, key=None, value=0):
        """Initialize an Entry with optional key and initial value.
        
        Args:
            key: The key to store (default: None)
            value: Initial value (default: 0)
        """
        self.key = key
        self.value = value
        # self.flag = False
    
    def is_empty(self):
        """Check if this entry is empty (has no key).
        
        Returns:
            bool: True if key is None, False otherwise
        """
        return self.key is None


class HeavyFilter:
    """A filter data structure for identifying heavy hitters in data streams.
    
    Uses a slot-based approach where each slot contains a guard entry and multiple
    regular entries to track frequently occurring keys.
    """
    
    def __init__(self, num_slots, num_per_slot, KEY_T_SIZE=13):
        """Initialize the HeavyFilter with specified parameters.
        
        Args:
            num_slots: Number of slots in the filter
            num_per_slot: Number of entries per slot
            KEY_T_SIZE: Size of the key in bytes (default: 13)
        """
        self.key_size = KEY_T_SIZE
        self.num_slots = calNextPrime(num_slots)
        self.slots = [[Entry() for _ in range(num_per_slot)] for _ in range(self.num_slots)]
        self.h, self.s, self.n = [GenHashSeed(i) for i in range(3)] 

    def hash(self, key):
        """Calculate the hash index for a given key.
        
        Args:
            key: The key to hash
            
        Returns:
            int: The slot index where this key should be stored
        """
        return AwareHash(key, self.key_size, self.h, self.s, self.n) % self.num_slots
    
    def get_memory_usage(self):
        """Calculate the approximate memory usage of this filter.
        
        Returns:
            float: Memory usage in bytes
        """
        return (len(self.slots) * len(self.slots[0]) * (4  + self.key_size))
    
    def insert(self, key, value=1, threshold=8):
        """Insert a key-value pair into the filter.
        
        Args:
            key: The key to insert
            value: The value to add (default: 1)
            threshold: Threshold for determining when to swap entries (default: 8)
            
        Returns:
            tuple or int: 
                0 - Key was inserted or updated in place
                1 - Guard value incremented but no swap occurred
                (-1, (swap_key, swap_value)) - Entry was swapped out
        """
        slot_index = self.hash(key)
        slot = self.slots[slot_index]

        # Iterate through non-guard entries (starting from index 1)
        # Using direct indexing instead of slicing for better performance
        for i in range(1, len(slot)):
            entry = slot[i]
            
            if entry.is_empty():
                entry.key = key
                entry.value += value
                return 0, None
            
            elif entry.key == key:
                entry.value += value
                return 0, None
        
        # Find minimum entry among non-guard entries
        min_entry = min(slot[1:], key=lambda e: e.value)
        guard_entry = slot[0]
        guard_entry.value += 1

        lambda_ = guard_entry.value / min_entry.value
        if lambda_ < threshold:
            return 1, None
        else:
            swap_key = min_entry.key
            swap_value = min_entry.value
            guard_entry.value = 0
            min_entry.key = key
            min_entry.value = value
            # min_entry.flag = True
            return -1, (swap_key, swap_value)
        
    def query(self, key):
        """Query the value associated with a key.
        
        Args:
            key: The key to look up
            
        Returns:
            int: value which is the count
        """
        slot_index = self.hash(key)
        slot = self.slots[slot_index]

        # Check non-guard entries for the key
        for i in range(1, len(slot)):
            entry = slot[i]
            if entry.key == key:
                return entry.value
            
        return 0