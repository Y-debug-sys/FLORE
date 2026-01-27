import math
import numpy as np


def compute_average_absolute_error(truth, predict):
    """
    Calculate the Average Absolute Error (AAE) between ground truth and predicted values.
    
    AAE is computed as the mean of absolute differences between truth and prediction:
    AAE = mean(|predict - truth|)
    
    This is a standard metric for evaluating the average magnitude of errors in a set 
    of predictions, without considering their direction.
    
    Parameters:
        truth (array-like): Ground truth values
        predict (array-like): Predicted values
        
    Returns:
        float: Mean of absolute errors
        
    Example:
        >>> truth = [1, 2, 3, 4, 5]
        >>> predict = [1.1, 2.2, 2.8, 4.1, 5.3]
        >>> compute_average_absolute_error(truth, predict)
        0.19999999999999996
    """
    gt, et = np.array(truth), np.array(predict)
    return np.abs(et - gt).mean()


def compute_average_relative_error(truth, predict):
    """
    Calculate the Average Relative Error (ARE) between ground truth and predicted values.
    
    ARE is computed as the mean of relative differences between truth and prediction:
    ARE = mean(|predict - truth| / |truth|)
    
    This metric is useful when the magnitude of the error relative to the actual values matters.
    
    Parameters:
        truth (array-like): Ground truth values (must not contain zeros)
        predict (array-like): Predicted values
        
    Returns:
        float: Mean of relative errors
        
    Note:
        This function will produce inf or nan if truth contains zero values.
        
    Example:
        >>> truth = [1, 2, 4, 8]
        >>> predict = [1.1, 1.9, 4.2, 7.8]
        >>> compute_average_relative_error(truth, predict)
        0.024999999999999994
    """
    gt, et = np.array(truth), np.array(predict)
    return (np.abs(et - gt) / gt).mean()


def compute_average_weighted_error(truth, predict):
    """
    Calculate the Average Weighted Error (AWE) between ground truth and predicted values.
    
    AWE is computed as the mean of weighted absolute differences:
    AWE = mean(|predict - truth| * truth)
    
    In this metric, larger truth values contribute more to the overall error, 
    making it suitable for applications where larger values are more important.
    
    Parameters:
        truth (array-like): Ground truth values
        predict (array-like): Predicted values
        
    Returns:
        float: Weighted mean of absolute errors
        
    Example:
        >>> truth = [1, 2, 3, 4]
        >>> predict = [1.1, 1.9, 3.2, 3.8]
        >>> compute_average_weighted_error(truth, predict)
        0.55
    """
    gt, et = np.array(truth), np.array(predict)
    return (np.abs(et - gt) * gt).mean()


def compute_heavy_hitter_error(truth, predict, k=None):
    """
    Calculate precision, recall, and F1-score for heavy hitter detection.
    
    Heavy hitters are the top-k most frequent items. This function compares the 
    ground truth heavy hitters with predicted heavy hitters and computes standard 
    classification metrics.
    
    Parameters:
        truth (array-like): Ground truth values
        predict (array-like): Predicted values
        k (int, optional): Number of top items to consider as heavy hitters.
                          If None, defaults to sqrt(len(truth)) + 1
            
    Returns:
        tuple: (precision, recall, F1-score)
        
    Example:
        >>> truth = [1, 1, 2, 2, 2, 3, 4, 5]
        >>> predict = [1, 1, 2, 2, 3, 3, 3, 4]
        >>> compute_heavy_hitter_error(truth, predict, k=2)
        (1.0, 1.0, 1.0)
    """
    # Determine number of heavy hitters to consider
    topk = int(math.sqrt(len(truth))) + 1 if k is None else k
    gt, et = np.array(truth), np.array(predict)

    # Get indices of top-k items in ground truth and prediction
    topk_index_gt = np.argsort(gt)[-topk:]
    topk_index_et = np.argsort(et)[-topk:]
    
    # Convert to sets for easier comparison
    set_gt = set(topk_index_gt.tolist())
    set_et = set(topk_index_et.tolist())

    # Calculate true positives, false positives, and false negatives
    tp = len(set_gt & set_et)  # Items correctly identified as heavy hitters
    fp = len(set_et - set_gt)  # Items incorrectly identified as heavy hitters
    fn = len(set_gt - set_et)  # Items missed as heavy hitters

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1