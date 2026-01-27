from Utils.sketch_utils import calNextPrime
from Baselines.classic_sketch.count import CountSketch
from Baselines.classic_sketch.count_min import CountMinSketch


class UniEntry:
    """
    A unified entry structure to hold key and count information in the augmented sketch.
    
    This class represents a single entry in the filter array, storing both the key
    and its associated counts (new and old) to support the augmented sketch operations.
    
    Attributes:
        key: The key stored in this entry, or None if empty
        key_size (int): Size of the key in bytes
        new_count (int): Current count value for the key
        old_count (int): Previous count value, used for tracking changes
    """

    def __init__(self, key, new_count=0, key_size=8):
        """
        Initialize a UniEntry instance.
        
        Args:
            key: The key to store in this entry
            new_count (int): Initial count value. Defaults to 0.
            key_size (int): Size of the key in bytes. Defaults to 8.
        """
        self.key = key
        self.key_size = key_size
        self.new_count = new_count
        self.old_count = 0

    @classmethod
    def from_key(cls, key, key_size=8):
        """
        Create a UniEntry instance from a key.
        
        Args:
            key: The key to store in the entry
            key_size (int): Size of the key in bytes. Defaults to 8.
            
        Returns:
            UniEntry: New instance with the specified key
        """
        return cls(key, key_size=key_size)

    @classmethod
    def default(cls, key_size=8):
        """
        Create a default (empty) UniEntry instance.
        
        Args:
            key_size (int): Size of the key in bytes. Defaults to 8.
            
        Returns:
            UniEntry: New instance with None key and zero counts
        """
        return cls(None, key_size=key_size)

    def get_memory_usage(self):
        """
        Calculate the memory usage of this entry.
        
        Returns:
            int: Memory usage in bytes (2 integers of 4 bytes each plus key size)
        """
        return 2 * 4 + self.key_size
    

class AugmentedSketch:
    """
    An augmented sketch implementation combining a filter array with a classic sketch.
    
    This sketch uses a two-tier approach:
    1. A filter array (UniEntry list) that stores exact counts for frequently occurring keys
    2. A backup classic sketch (CountMin or CountSketch) for handling overflow
    
    The augmentation strategy improves accuracy by maintaining exact counts for popular items
    in the filter while delegating rare items to the underlying sketch structure.
    
    Attributes:
        key_size (int): Size of keys in bytes
        length (int): Length of the filter array (rounded to next prime)
        filters (list): Array of UniEntry objects serving as the exact filter
        sketch: Underlying sketch structure (CountMinSketch or CountSketch)
    """

    def __init__(self, num_entries, width, depth, KEY_T_SIZE=13, sketch='cm'):
        """
        Initialize the AwareHashSketch.
        
        Args:
            num_entries (int): Number of entries in the filter array
            width (int): Width of the underlying sketch
            depth (int): Depth of the underlying sketch
            KEY_T_SIZE (int): Size of keys in bytes. Defaults to 13.
            sketch (str): Type of underlying sketch ('cm' for CountMin, 'cs' for CountSketch).
                         Defaults to 'cm'.
                         
        Raises:
            ValueError: If sketch parameter is not 'cm' or 'cs'
        """
        self.key_size = KEY_T_SIZE
        self.length = calNextPrime(num_entries)
        self.filters = [UniEntry.default(KEY_T_SIZE) for _ in range(self.length)]

        if sketch == 'cs':
            self.sketch = CountSketch(width, depth, KEY_T_SIZE)
        elif sketch == 'cm':
            self.sketch = CountMinSketch(width, depth, KEY_T_SIZE)
        else:
            raise ValueError("Invalid sketch type. Choose 'cs' or 'cm'.")
        
    def insert(self, key, value=1):
        """
        Insert a key-value pair into the sketch.
        
        The insertion logic follows these steps:
        1. Look up the key in the filter array
        2. If found, increment its count
        3. If not found but there's an empty slot, store it there
        4. Otherwise, delegate to the underlying sketch and potentially evict the 
           least frequent item from the filter to the sketch
        
        Args:
            key: The key to insert
            value (int): The value to add to the key's count. Defaults to 1.
        """
        flag, idx_or_min = self.lookup_key(key)

        if flag == 0:
            # Key not found, empty slot available
            self.filters[idx_or_min].key = key
            self.filters[idx_or_min].new_count = value
        
        elif flag == 1:
            # Key found in filter, increment count
            self.filters[idx_or_min].new_count += value

        else:
            # Filter full, use underlying sketch
            self.sketch.insert(key, value)
            estimated_count = self.sketch.query(key)

            # If the sketch estimate is higher than the minimum count in filter,
            # evict the minimum entry and replace with the new key
            if estimated_count > idx_or_min:
                idx = self.lookup_min(idx_or_min)
                exchanged_value = self.filters[idx].new_count - self.filters[idx].old_count

                if exchanged_value > 0:
                    exchanged_key = self.filters[idx].key
                    self.sketch.insert(exchanged_key, exchanged_value)

                self.filters[idx].key = key
                self.filters[idx].new_count = value
                self.filters[idx].old_count = value

    def lookup_key(self, key):
        """
        Look up a key in the filter array.
        
        Args:
            key: The key to look up
            
        Returns:
            tuple: (flag, index_or_min_value) where:
                   - flag=0: Key not found, empty slot at index
                   - flag=1: Key found at index
                   - flag=-1: Key not found, return minimum count value
                   - index_or_min_value: Either the index or minimum count depending on flag
        """
        min_value = float('inf')
        for idx, entry in enumerate(self.filters):
            if entry.key == key:
                return 1, idx
            
            if entry.key is None:
                return 0, idx
            
            min_value = min(min_value, entry.new_count)
        
        return -1, min_value
    
    def lookup_min(self, min_value):
        """
        Find the index of an entry with the specified minimum count value.
        
        Args:
            min_value: The count value to search for
            
        Returns:
            int: Index of the entry with the specified count value
            
        Raises:
            ValueError: If no entry with the specified value is found
        """
        for idx, entry in enumerate(self.filters):
            if entry.new_count == min_value:
                return idx
        
        raise ValueError("Min value not found in the filters.")

    def query(self, key):
        """
        Query the count estimate for a key.
        
        Args:
            key: The key to query
            
        Returns:
            int: Estimated count of the key
        """
        flag, idx_or_min = self.lookup_key(key)

        if flag == 1:
            # Key found in filter, return exact count
            return self.filters[idx_or_min].new_count
        else:
            # Key not in filter, query underlying sketch
            return self.sketch.query(key)

    def get_memory_usage(self):
        """
        Calculate the total memory usage of the sketch.
        
        Returns:
            int: Total memory usage in bytes
        """
        return self.length * UniEntry.default(key_size=self.key_size).get_memory_usage() +\
               self.sketch.get_memory_usage()