import math
from collections import Counter


def compute_weighted_mean_relative_difference(truth, predict):
    """
    Calculate the Weighted Mean Relative Difference (WMRD) between ground truth and predicted values.
    
    WMRD is computed as the sum of absolute differences normalized by the sum of averages:
    WMRD = Σ|gt_i - pred_i| / Σ((gt_i + pred_i) / 2)
    
    This metric is particularly useful for evaluating frequency estimation accuracy where
    larger counts contribute more to the overall error.
    
    Parameters:
        truth (list or array-like): Ground truth values
        predict (list or array-like): Predicted values
        
    Returns:
        float: Weighted mean relative difference between truth and predict
        
    Example:
        >>> truth = [1, 2, 2, 3, 3, 3]
        >>> predict = [1, 2, 2, 2, 3, 3] 
        >>> weighted_mean_relative_difference(truth, predict)
        0.142857...
    """
    # Count occurrences of each item in truth and prediction
    gt_count = dict(Counter(truth))
    et_count = dict(Counter(predict))
    
    # Get all unique items from both truth and prediction
    union_count = set(gt_count.keys()).union(set(et_count.keys()))
    
    numerator_sum = 0
    denominator_sum = 0
    
    # For each unique item, calculate contribution to WMRD
    for item in union_count:
        # Get count in ground truth (default to 0 if not present)
        count_truth = gt_count.get(item, 0)
        
        # Get count in prediction (default to 0 if not present)
        count_predict = et_count.get(item, 0)
        
        # Accumulate the absolute difference and average
        numerator_sum += abs(count_truth - count_predict)
        denominator_sum += (count_truth + count_predict) / 2
    
    # Avoid division by zero
    if denominator_sum == 0:
        return 0.0
    
    return numerator_sum / denominator_sum


def compute_entropy_absolute_error(truth, predict):
    """
    Calculate the absolute difference between entropies of ground truth and predicted values.
    
    Entropy is calculated as: H(X) = Σ p(x) * log2(1/p(x))
    In this case, it's computed as: H(X) = Σ i * (c/n) * log2(n/c)
    where i is the item value, c is its count, and n is total number of items.
    
    This metric evaluates how well the prediction preserves the entropy (randomness) 
    of the original data distribution.
    
    Parameters:
        truth (list or array-like): Ground truth values
        predict (list or array-like): Predicted values
        
    Returns:
        float: Absolute difference between entropies of truth and predict
        
    Example:
        >>> truth = [1, 2, 2, 3, 3, 3]
        >>> predict = [1, 1, 2, 3, 3, 3]
        >>> entropy_absolute_error(truth, predict)
        0.23456...
    """
    # Total number of items
    n_key = len(truth)
    
    # Handle edge case
    if n_key == 0:
        return 0.0
    
    # Initialize entropy accumulators
    gt_entropy = 0.0
    et_entropy = 0.0
    
    # Count occurrences in truth and prediction
    gt_count = dict(Counter(truth))
    et_count = dict(Counter(predict))
    
    # Calculate entropy of ground truth
    for item_value, count in gt_count.items():
        # Skip items with zero count or zero value to avoid log(0)
        if count > 0 and item_value > 0:
            probability = count / n_key
            gt_entropy += item_value * probability * math.log2(1 / probability)
    
    # Calculate entropy of predictions
    for item_value, count in et_count.items():
        # Skip items with zero count or zero value to avoid log(0)
        if count > 0 and item_value > 0:
            probability = count / n_key
            et_entropy += item_value * probability * math.log2(1 / probability)
    
    return abs(et_entropy - gt_entropy)
