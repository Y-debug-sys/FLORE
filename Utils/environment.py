import time
# import math
import random
import numpy as np

from tqdm.auto import tqdm
from Utils.data_utils import gen_stream, read_items, read_traces


class Env(object):
    """
    Base environment class for sketch algorithm testing.
    
    This class provides the basic framework for inserting items into various sketch data structures
    and sampling from them at regular intervals.
    
    Args:
        args: Arguments containing configuration parameters
        logger (logging.Logger): Logger instance for recording training information
        sketches (dict): Dictionary of sketch objects to be tested
    """
    
    def __init__(self, args, logger, sketches: dict):
        """
        Initialize the environment with sketches, logger, and configuration parameters.
        
        Args:
            args: Configuration arguments containing key_size, seed, interval, and other parameters
            logger: Logger instance for recording information
            sketches: Dictionary of sketch objects to be tested
            
        Raises:
            KeyError: If 'flore' sketch is not found in the sketches dictionary
        """
        super(Env, self).__init__()

        try: 
            depth, width = sketches['flore'].depth, sketches['flore'].width
        except KeyError: 
            raise KeyError("FLORE not found in the sketches")

        self.ground_truth = {}
        self.record_truth = {}
        self.key_size, self.window = args.key_size, args.window
        self.samples, self.logger = np.empty([0, depth, width]), logger
        self.sketches, self.seed, self.interval = sketches, args.seed, args.interval

    def _insert_sketches(self, item, num_insert=1, sketch_names=None):
        """
        Insert an item into all sketch data structures.
        
        Args:
            item: The item to be inserted into sketches
            num_insert (int): Number of times to insert the item, defaults to 1
            sketch_names (list, optional): List of sketch names to insert into. Defaults to None.
        """
        if sketch_names is not None:
            for sketch_name in sketch_names:
                self.sketches[sketch_name].insert(item, num_insert)
        else: 
            for sketch in self.sketches.values():
                sketch.insert(item, num_insert)

    def _update_gt(self, item, num_insert=1):
        """
        Update the ground truth count for a given item.
        
        This function increments the count of an item in the ground truth dictionary.
        If the item already exists, its count is increased by num_insert. Otherwise,
        the item is added to the dictionary with an initial count of num_insert.
        
        Args:
            item: The item whose count needs to be updated
            num_insert (int): The amount to increment the item's count by, defaults to 1
        """
        if item in self.ground_truth:
            self.ground_truth[item] += num_insert
        else:
            self.ground_truth[item] = num_insert

    def _sample_sketch(self, sketch): 
        """
        Sample the current state of a sketch and store it.
        
        Args:
            sketch: The sketch object to sample from
        """
        sample = np.expand_dims(sketch.get_counters(), axis=0)
        self.samples = np.row_stack([self.samples, sample])

    def _load_data(self):
        """
        Abstract method for loading data streams.
        
        This method should be implemented by subclasses to provide specific data loading logic.
        
        Returns:
            None: This base implementation raises NotImplementedError
        """
        raise NotImplementedError
        return None

    def run(self, start_point=0, break_point=None):
        """
        Execute the main loop for processing the data stream.

        Args:
            start_point (int): The starting point for processing the data stream, defaults to 0
            break_point (int, optional): The index to stop processing the data stream. 
                                       Defaults to None, which means it will use self.break_point.
        
        This method loads the data stream, inserts items into sketches at regular intervals,
        and samples the sketch states for analysis.
        """
        item_count = 0
        size, stream = self._load_data()
        self.logger.info('Inserting...')
        if break_point is None: break_point = self.break_point
        stream = stream[start_point:break_point] if break_point < size else stream
        with tqdm(initial=0, total=break_point-start_point, desc=f'Inserting stream into the ({len(self.sketches)}) sketches') as pbar:
            for idx, item in enumerate(stream):
                self._insert_sketches(item)
                self._update_gt(item)
                item_count += 1
                pbar.update(1)

                if idx > self.interval and idx % self.interval == 0:
                    self._sample_sketch(self.sketches['flore'])
        
        self.logger.info(f'Inserted {item_count} items with {len(self.ground_truth)} distinct keys.')

    def run_fast(self, start_point=0, break_point=None):
        """
        Run the direct insertion process for sketch algorithms more quickly.
        
        This function first reads the streaming data and updates the ground truth,
        then inserts all items into the sketches. It differs from the standard run
        method by batching insertions based on item frequencies.
        
        Args:
            start_point (int): The index to start processing the data stream from. Defaults to 0.
            break_point (int, optional): The index to stop processing the data stream. 
                                       Defaults to None, which means it will use self.break_point.
        
        Returns:
            None
        """
        item_count = 0
        size, stream = self._load_data()
        self.logger.info('Inserting...')
        if break_point is None: break_point = self.break_point
        stream = stream[start_point:break_point] if break_point < size else stream
        
        # Process streaming data and update ground truth
        with tqdm(initial=0, total=break_point-start_point, desc='Reading streaming data') as pbar:
            for idx, item in enumerate(stream):
                self._insert_sketches(item, sketch_names=['flore'])
                self._update_gt(item)
                item_count += 1
                pbar.update(1)

                if idx > self.interval and idx % self.interval == 0:
                    self._sample_sketch(self.sketches['flore'])

        new_sketch_names = [x for x in self.sketches.keys() if x != 'flore']
        
        # Insert all items into sketches with their respective volumes
        with tqdm(initial=0, total=len(self.ground_truth), desc=f'Inserting data into the ({len(self.sketches)}) sketches') as pbar:
            for key in self.ground_truth.keys():
                volume = self.ground_truth[key]
                self._insert_sketches(key, volume, sketch_names=new_sketch_names)
                pbar.update(1)
        
        self.logger.info(f'Inserted {item_count} items with {len(self.ground_truth)} distinct keys.')

    def recover(self, nn_solver=None, is_clear=True):
        """
        Recover the frequency estimation results from sketches and ground truth.
        
        This function queries all sketches with keys from ground truth to get their estimated 
        frequency values, and optionally applies neural network solver for FLORE sketch.
        
        Args:
            nn_solver: Neural network solver for decoding FLORE sketch results, only applicable 
                      for FLORE with non-local decoding. Defaults to None.
            is_clear (bool): Whether to clear the decoding results in FLORE sketch after recovery. 
                           Defaults to True.
        
        Returns:
            tuple: A tuple containing:
                - recovery_results (np.ndarray): Array of recovered frequency values 
                - sketch_names (list): List of sketch names with 'gt' appended at the end 
        """
        self.logger.info('Recovering...')
        recovery_results = [[] for _ in range(len(self.sketches) + 1)]
        if nn_solver is None: 
            assert self.sketches['flore'].decoding_method == 'local', "NN solver is only applicable for FLORE with non-local decoding" 
        
        # Apply neural network solver for FLORE sketch if provided
        if nn_solver is not None:
            counters = self.sketches['flore'].get_counters()
            pred_vec = nn_solver.test_in_sample(counters)
            self.sketches['flore'].set_decoding_results(pred_vec)

        # Query all sketches for each key in ground truth
        for key in self.ground_truth.keys(): 
            volume = self.ground_truth[key]
            recovery_results[-1].append(volume)
            for idx, sketch in enumerate(self.sketches.values()): 
                volume = sketch.query(key)
                recovery_results[idx].append(volume)
        
        # Clear decoding results if required
        if is_clear: 
            self.sketches['flore'].set_decoding_results(None)
            try:
                self.sketches['nze'].clear()
            except:
                pass
            try:
                self.sketches['pr'].clear()
            except:
                pass

        sketch_names = list(self.sketches.keys())
        sketch_names.append('gt')
        self.logger.info(f'Recovered {len(self.ground_truth)} keys from {len(self.sketches)} sketches.')
        return np.asarray(recovery_results), sketch_names

    def get_processing_time(self, test_range=5000, is_random=False):
        """
        Calculate the average processing time for sketch insert operations.
        
        This method measures the time it takes to insert a specified number of items
        into each sketch and calculates the average insertion time.
        
        Args:
            test_range (int): Number of items to test with, defaults to 5000
            is_random (bool): Whether to select random samples or use first N items, defaults to False
            
        Returns:
            dict: A dictionary mapping sketch names to their average insertion time
        """
        average_time = {}
        _, stream = self._load_data()

        if is_random:
            samples = random.sample(stream, test_range)
        else:
            samples = stream[:test_range]
        
        for sname in self.sketches.keys():
            if sname[0] == 'l': self.sketches[sname].set_mode(True)

        # Fixed: iterate over self.sketches.items() instead of self.sketches 
        for sketch_name, sketch in self.sketches.items():
            start = time.perf_counter()

            for _, item in enumerate(samples):
                sketch.insert(item)

            end = time.perf_counter()
            average_time[sketch_name] = (end-start) / test_range

        for sname in self.sketches.keys():
            if sname[0] == 'l': self.sketches[sname].set_mode(False)

        return average_time
    
    def get_samples_in_window(self, window: int = None):
        """
        Retrieve the most recent samples within the specified window size.
        
        This method returns a slice of the stored samples array, specifically the last 'window' 
        number of samples. If no window size is provided, it uses the default window size stored 
        in the instance.
        
        Args:
            window (int, optional): The number of recent samples to retrieve. If None, uses 
                                  the default window size from self.window.
                                  
        Returns:
            numpy.ndarray: A 3D array containing the samples within the specified window. 
                         The shape is (window, depth, width) where depth and width correspond 
                         to the sketch dimensions.
        """
        if window is None: window = self.window
        return self.samples[-window:, :, :]


class TraceEnv(Env):
    """
    Environment for processing trace data files.
    
    This environment loads data from trace files for sketch algorithm evaluation.
    
    Args:
        args: Arguments containing configuration parameters including data_path
        logger: Logger instance for recording information
        sketches (dict): Dictionary of sketch objects to be tested
    """
    def __init__(self, args, logger, sketches):
        """
        Initialize TraceEnv with data file path from args.
        
        Args:
            args: Configuration arguments containing data_fname
            logger: Logger instance for recording information
            sketches: Dictionary of sketch objects to be tested
        """
        super(TraceEnv, self).__init__(args, logger, sketches)
        self.data_path = args.data_fname

    def _load_data(self):
        """
        Load data from trace files.
        
        Returns:
            tuple: A tuple containing:
                - size (int): The number of traces read from the data file
                - stream (iterable): The data stream read from the file
        """
        size, stream = read_traces(self.data_path, self.key_size)
        self.logger.info(f"Read {size} traces from {self.data_path}.")
        return size, stream


class DataEnv(Env):
    """
    Environment for processing general data files.
    
    This environment loads data from general data files for sketch algorithm evaluation.
    
    Args:
        args: Arguments containing configuration parameters including data_path
        logger: Logger instance for recording information
        sketches (dict): Dictionary of sketch objects to be tested
    """
    def __init__(self, args, logger, sketches):
        """
        Initialize DataEnv with data file path from args.
        
        Args:
            args: Configuration arguments containing data_fname
            logger: Logger instance for recording information
            sketches: Dictionary of sketch objects to be tested
        """
        super(DataEnv, self).__init__(args, logger, sketches)
        self.data_path = args.data_fname

    def _load_data(self):
        """
        Load data from general data files.
        
        Returns:
            tuple: A tuple containing:
                - size (int): The number of items read from the data file
                - stream (iterable): The data stream read from the file
        """
        size, stream = read_items(self.data_path, self.key_size)
        self.logger.info(f"Read {size} items from {self.data_path}.")
        return size, stream


class GenEnv(Env):
    """
    Environment for generating synthetic data streams.
    
    This environment generates synthetic data streams for sketch algorithm evaluation.
    
    Args:
        args: Arguments containing configuration parameters including num_samples and distribution
        logger: Logger instance for recording information
        sketches (dict): Dictionary of sketch objects to be tested
    """
    def __init__(self, args, logger, sketches):
        """
        Initialize GenEnv with synthetic data generation parameters from args.
        
        Args:
            args: Configuration arguments containing samples and distribution
            logger: Logger instance for recording information
            sketches: Dictionary of sketch objects to be tested
        """
        super(GenEnv, self).__init__(args, logger, sketches)
        self.num_samples, self.type = args.samples, args.distribution

    def _load_data(self):
        """
        Generate synthetic data stream based on specified distribution.
        
        Returns:
            tuple: A tuple containing:
                - size (int): Number of items generated
                - stream (list): List of generated items, each represented as bytes
        """
        size, stream =  gen_stream(self.num_samples, self.key_size, seed=self.seed, type=self.type)  # Fixed: use self.type instead of self.distribution
        self.logger.info(f"Synthsis {size} items sampled from {self.type} distribution.")  # Fixed: use self.type instead of self.distribution
        return size, stream