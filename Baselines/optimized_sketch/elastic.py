"""
Elastic Sketch Implementation

This module implements the Elastic Sketch data structure, which combines a heavy hitter 
detection component (using buckets with entries) with a light tail estimation component 
(using Count-Min Sketch). The elastic sketch dynamically manages memory between heavy 
and light parts based on traffic patterns.

The key innovation is the swapping mechanism: when a bucket is full, the algorithm decides 
whether to evict the minimum value entry to make room for a new heavy hitter, or to treat 
the new item as part of the light tail.
"""

from Baselines.classic_sketch.count_min import CountMinSketch
from Utils.sketch_utils import GenHashSeed, AwareHash, calNextPrime


def judge_if_swap(min_val, guard_val):
    """
    Determine whether to swap the minimum entry with a new flow.
    
    Args:
        min_val (int): Value of the minimum entry in the bucket
        guard_val (int): Current guard counter value
        
    Returns:
        bool: True if swap should occur (guard_val > 8 * min_val), False otherwise
        
    The condition uses bit shifting (min_val << 3) which is equivalent to min_val * 8.
    This ensures that we only swap when the guard counter indicates sufficient evidence
    that the new flow might be more significant than the current minimum.
    """
    return guard_val > (min_val << 3)


class Entry:
    """
    Represents an entry in the heavy part buckets of the Elastic Sketch.
    
    Each entry stores a flow key, its accumulated value, and a flag indicating
    whether it has been swapped out from the heavy part to the light part.
    """

    def __init__(self, flowkey=None, val=0):
        """Initialize an empty entry or with specific flow key and value."""
        self.flowkey = flowkey  # The flow identifier (None indicates empty slot)
        self.val = val          # Accumulated value/count for this flow
        self.flag = False       # True if this entry was swapped out to light part
    
    def is_empty(self):
        """Check if this entry slot is empty (no flow key assigned)."""
        return self.flowkey is None


class ElasticSketch:
    """
    Elastic Sketch Data Structure
    
    Combines two components:
    1. Heavy Part: Array of buckets containing Entry objects for potential heavy hitters
    2. Light Part: Count-Min Sketch for estimating frequencies of non-heavy flows
    
    The structure dynamically manages memory allocation between heavy and light parts
    based on observed traffic patterns, making it "elastic" in its resource usage.
    """

    def __init__(self, num_buckets, num_per_bucket, width, depth, KEY_T_SIZE=13):
        """
        Initialize the Elastic Sketch.
        
        Args:
            num_buckets (int): Number of buckets in the heavy part
            num_per_bucket (int): Number of entries per bucket (last entry is guard)
            width (int): Width of the Count-Min sketch (light part)
            depth (int): Depth of the Count-Min sketch (light part)
            KEY_T_SIZE (int): Size of the flow key in bytes (default: 13)
        """
        self.key_size = KEY_T_SIZE
        self.num_buckets = calNextPrime(num_buckets)  # Ensure prime number for better hashing
        self.num_per_bucket = num_per_bucket
        self.cm = CountMinSketch(width, depth, KEY_T_SIZE)  # Light part component
        # Initialize buckets: each bucket has num_per_bucket entries
        self.buckets = [[Entry() for _ in range(num_per_bucket)] for _ in range(self.num_buckets)]
        # Generate three hash seeds for different hashing purposes
        self.h, self.s, self.n = [GenHashSeed(i) for i in range(3)]

    def hash(self, key):
        """
        Hash a flow key to determine its bucket index in the heavy part.
        
        Args:
            key: Flow key to hash
            
        Returns:
            int: Bucket index (0 to num_buckets-1)
        """
        return AwareHash(key, self.key_size, self.h, self.s, self.n) % self.num_buckets
    
    def get_memory_usage(self):
        """
        Calculate approximate memory usage of the Elastic Sketch.
        
        Returns:
            float: Memory usage in bytes
            
        Calculation includes:
        - Heavy part: buckets * entries_per_bucket * (4 bytes for val + key_size + 0.125 for flag)
        - Light part: memory from Count-Min sketch
        """
        return (len(self.buckets) * len(self.buckets[0]) * (4  + self.key_size + 0.125) + self.cm.get_memory_usage())

    def heavypart_insert(self, flowkey, val):
        """
        Insert a flow into the heavy part (buckets).
        
        Args:
            flowkey: Flow identifier
            val (int): Value to add (typically 1 for packet counting)
            
        Returns:
            int or tuple: 
                - 0: Successfully inserted into existing or empty slot
                - 2: Bucket full, no swap performed (flow should go to light part)
                - (1, swap_key, swap_val): Swap performed, returns evicted flow info
        """
        # matched, empty = -1, -1  # Unused variables (commented out in original)
        index = self.hash(flowkey)
        bucket = self.buckets[index]
        
        # Check first (num_per_bucket - 1) entries for match or empty slot
        for i, entry in enumerate(bucket[:-1]):
            if entry.flowkey == flowkey:
                # Found existing flow, increment its value
                entry.val += val
                return 0

            if entry.is_empty():
                # Found empty slot, insert new flow
                entry.flowkey = flowkey
                entry.val = val
                return 0

        # Bucket is full - handle eviction decision
        min_entry = min(bucket[:-1], key=lambda e: e.val)  # Find entry with minimum value
        guard_entry = bucket[-1]  # Last entry in bucket serves as guard counter
        guard_entry.val += 1      # Increment guard counter

        # Decide whether to swap based on guard vs minimum value ratio
        if not judge_if_swap(min_entry.val, guard_entry.val):
            return 2  # No swap, flow should be handled by light part
        else:
            # Perform swap: evict minimum entry, insert new flow
            swap_key = min_entry.flowkey
            swap_val = min_entry.val
            guard_entry.val = 0           # Reset guard counter
            min_entry.flowkey = flowkey   # Insert new flow
            min_entry.val = val
            min_entry.flag = True         # Mark as swapped
            return 1, swap_key, swap_val  # Return evicted flow info

    def lightpart_insert(self, flowkey, val):
        """
        Insert a flow into the light part (Count-Min Sketch).
        
        Args:
            flowkey: Flow identifier
            val (int): Value to add
        """
        self.cm.insert(flowkey, val)

    def insert(self, flowkey, val=1):
        """
        Main insertion method for the Elastic Sketch.
        
        Routes flows between heavy and light parts based on heavypart_insert results.
        
        Args:
            flowkey: Flow identifier
            val (int): Value to add (default: 1)
        """
        result = self.heavypart_insert(flowkey, val)
        if result == 0:
            # Successfully handled by heavy part
            return
        elif result == 2:
            # Heavy part full, no swap - handle by light part
            self.lightpart_insert(flowkey, val)
        elif result[0] == 1:
            # Swap occurred - insert evicted flow into light part
            swap_key, swap_val = result[1], result[2]
            self.lightpart_insert(swap_key, swap_val)

    def heavypart_query(self, flowkey):
        """
        Query the heavy part for a flow's value and swap status.
        
        Args:
            flowkey: Flow identifier to query
            
        Returns:
            tuple: (value, flag) where value is the accumulated count and 
                   flag indicates if the entry was swapped out
        """
        index = self.hash(flowkey)
        for entry in self.buckets[index][:-1]:
            if entry.flowkey == flowkey:
                return entry.val, entry.flag
        return 0, False

    def lightpart_query(self, flowkey):
        """
        Query the light part (Count-Min Sketch) for a flow's estimated value.
        
        Args:
            flowkey: Flow identifier to query
            
        Returns:
            int: Estimated value from Count-Min Sketch
        """
        return self.cm.query(flowkey)

    def query(self, flowkey):
        """
        Main query method for the Elastic Sketch.
        
        Combines results from heavy and light parts according to the elastic sketch logic:
        - If flow found in heavy part and NOT flagged as swapped, return heavy part value only
        - If flow found in heavy part but flagged as swapped, add light part estimate
        - If flow not found in heavy part, return light part estimate
        
        Args:
            flowkey: Flow identifier to query
            
        Returns:
            int: Estimated total value for the flow
        """
        heavy_result, flag = self.heavypart_query(flowkey)
        if heavy_result == 0 or flag:
            # Flow not in heavy part, or was swapped out - include light part estimate
            light_result = self.lightpart_query(flowkey)
        else:
            # Flow is active in heavy part - don't double count with light part
            light_result = 0
        return heavy_result + light_result