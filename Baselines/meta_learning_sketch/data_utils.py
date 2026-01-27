# =============================================================================================================
# Source / Reference:
#   - Yuan Feng and Yukun Cao
#   - https://github.com/FFY0/LegoSketch_ICML/blob/main/SourceCode/TaskRelatedClasses/SupportGenerator.py
#
# License:
#   <License type, e.g., MIT License / Apache 2.0 / GPL>
#   <Optional: link to license text>
# =============================================================================================================


import torch
import random
import numpy as np

# from torch.utils.data import Dataset, DataLoader


def train_loader(data_generator):
    while True:
        support_x, support_y = data_generator.sample_training_data()
        query_x, query_y = support_x.clone(), support_y.clone().reshape(-1, 1)
        yield (support_x.float(), support_y.float(), query_x.float(), query_y.float())


class SimpleZipfGeneratorByIS:
    """
    A generator for Zipf-distributed data using Importance Sampling.
    
    This class generates Zipf-distributed data samples with configurable parameters
    for training meta-learning algorithms. It maintains a large pool of samples
    and provides them in batches according to Zipf distributions with varying parameters.
    """

    def __init__(
        self,
        scale_upper, 
        scale_lower,
        skew_upper=10,
        skew_lower=0.5,
        zipf_scale_upper=0.8, 
        zipf_scale_lower=1.3,
        sample_IS_scale='linear',
        sample_size=1_000_000,
        device=None
    ):
        """
        Initialize the Zipf generator with specified parameters.
        
        Args:
            scale_upper (int): Upper bound for number of items in distribution
            scale_lower (int): Lower bound for number of items in distribution
            skew_upper (float): Upper bound for skew ratio (default: 10)
            skew_lower (float): Lower bound for skew ratio (default: 0.5)
            zipf_scale_upper (float): Upper bound for Zipf distribution parameter (default: 0.8)
            zipf_scale_lower (float): Lower bound for Zipf distribution parameter (default: 1.3)
            sample_IS_scale (str): Sampling method, either 'linear' or 'log' (default: 'linear')
            sample_size (int): Size of the sample pool (default: 1,000,000)
            device (torch.device): Computation device (default: CUDA if available, otherwise CPU)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.sample_IS_func = self.sample_by_linear_scale if sample_IS_scale == "linear" else self.sample_by_log_scale
        self.scale_upper, self.scale_lower = scale_upper, scale_lower 
        self.skew_upper, self.skew_lower = skew_upper, skew_lower 
        self.zipf_scale_upper, self.zipf_scale_lower = zipf_scale_upper, zipf_scale_lower

        self.num_of_generated_task = 0
        self.sample_size, self.sample_index = sample_size, 0
        self.samples_ndarray = np.arange(sample_size).reshape(-1, 1)
        self.samples_tensor = torch.tensor(self.samples_ndarray, device=self.device).float()
        self.shuffle()

    def shuffle(self):
        """Randomly shuffle the sample tensor to ensure randomness in sampling."""
        index = torch.randperm(self.samples_tensor.shape[0], device=self.device)
        self.samples_tensor = self.samples_tensor[index]

    def reset_index(self, index=0):
        """
        Reset the sample index to a specified position.
        
        Args:
            index (int): Index position to reset to (default: 0)
        """
        self.sample_index = index

    def sample_by_log_scale(self, upper_ratio=1.0):
        """
        Sample number of items using logarithmic scaling.
        
        Args:
            upper_ratio (float): Ratio to adjust the upper bound (default: 1.0)
            
        Returns:
            int: Number of items sampled
        """
        uniform_ratio = random.random() * upper_ratio
        num_items = int(((self.scale_upper / self.scale_lower) ** uniform_ratio) * self.scale_lower)
        return num_items

    def sample_by_linear_scale(self, upper_ratio=1.0):
        """
        Sample number of items using linear scaling.
        
        Args:
            upper_ratio (float): Ratio to adjust the upper bound (default: 1.0)
            
        Returns:
            int: Number of items sampled
        """
        uniform_ratio = random.random() * upper_ratio
        num_items = int(uniform_ratio * (self.scale_upper - self.scale_lower) + self.scale_lower)
        return num_items

    def get_zipf_data(self, zipf_scale, num_items):
        """
        Generate Zipf-distributed weights for a given number of items.
        
        Args:
            zipf_scale (float): Parameter for the Zipf distribution
            num_items (int): Number of items in the distribution
            
        Returns:
            torch.Tensor: Vector of Zipf-distributed weights
        """
        # Create a decreasing sequence for Zipf distribution
        x = torch.arange(num_items, 0, step=-1, device=self.device).float()
        # Apply Zipf power law
        x = x ** (- zipf_scale)
        # Normalize to create probability distribution
        x = x / x.sum()
        # Scale by inverse of first element (max element)
        vector = x * (1 / x[0])
        # Randomly permute the vector
        index = torch.randperm(vector.shape[0], device=self.device)
        vector = vector[index]
        return vector

    def sample_training_data(self):
        """
        Sample a batch of training data with Zipf-distributed weights.
        
        Returns:
            tuple: (samples, weights) where samples is a tensor of item IDs and 
                   weights is a tensor of corresponding Zipf-distributed weights
        """
        # Sample the number of items for this batch
        num_items = self.sample_IS_func()
        
        # Sample skew ratio and Zipf distribution parameter
        skew_ratio = (random.random() * (self.skew_upper - self.skew_lower) + self.skew_lower)
        zipf_scale = (random.random() * (self.zipf_scale_upper - self.zipf_scale_lower) + self.zipf_scale_lower)
        
        # Generate Zipf-distributed weights
        weights = self.get_zipf_data(zipf_scale=zipf_scale, num_items=num_items) * skew_ratio

        # Check if we need to extend our sample pool
        if num_items + self.sample_index >= self.sample_size:
            self.shuffle()
            self.reset_index()

            while num_items + self.sample_index >= self.sample_size:
                # Extend the tensor by adding offset values to differentiate new samples
                extend_tensor = self.samples_tensor + self.samples_tensor.max() + 10
                self.samples_tensor = torch.cat([self.samples_tensor, extend_tensor], dim=0)
                # print('extend samples_tensor!', "original size", extend_tensor.shape[0],
                #       "new size", self.samples_tensor.shape[0])

        # Extract the samples for this batch
        samples = self.samples_tensor[self.sample_index: (num_items + self.sample_index)]
        self.reset_index(num_items + self.sample_index)
        self.num_of_generated_task += 1

        return samples, weights
