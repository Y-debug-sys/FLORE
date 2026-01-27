import torch
# import numpy as np

def expectation_maximization(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    A: torch.Tensor, 
    steps: int = 3,
    eps: float = 1e-8
):
    """
    Performs Expectation-Maximization algorithm to optimize input vector X 
    under given constraints to better fit the observed data (sketch counters) Y.
    
    This function iteratively updates x to minimize the error between 
    sketching operation A @ x and observations Y.
    
    Parameters:
        X (torch.Tensor): Input tensor of shape (batch_size, n_features), serves as initial solution
        Y (torch.Tensor): Observed data tensor of shape (batch_size, m_measurements)
        A (torch.Tensor): Transformation matrix of shape (m_measurements, n_features)
        steps (int): Number of iteration steps, default is 3
        eps (float): Numerical stability constant to prevent division by zero, default is 1e-8
    
    Returns:
        torch.Tensor: Optimized X
    """
    def linear_sketch(A, x):
        # Linear transformation: compute A @ x for each sample in batch
        return (A @ x.transpose(0, 1)).transpose(0, 1)

    # Initialize parameters and variables
    batch_size, device = X.shape[0], Y.device
    x = X.clone().detach()  # Current solution being optimized

    indexes = torch.arange(0, batch_size).to(device)  # Indexes for batch samples
    loss_min = torch.abs(linear_sketch(A, X) - Y).sum(dim=-1)  # Initial loss for each sample

    # EM algorithm iterative optimization process
    for _ in range(steps):
        # Compute update coefficients
        coeff1 = A / linear_sketch(A, x).unsqueeze(-1).clamp_min_(eps)  # Normalization factor
        coeff2 = x / A.sum(dim=0).unsqueeze(0).clamp_min_(eps)  # Prior probability factor

        # Calculate new x values
        x_coeff = torch.einsum('Bb,Bba->Ba', Y, coeff1)  # Weighted combination of measurements
        x_new = torch.mul(coeff2, x_coeff)  # Element-wise multiplication to get updated x

        # Evaluate quality of new solution and update best solution
        loss = torch.abs(linear_sketch(A, x_new) - Y).sum(dim=-1)  # Compute new loss
        index = indexes[(loss < loss_min).reshape(loss.shape)]  # Find improved samples
        
        # Update solutions for samples with improved loss
        if len(index) != 0:
            loss_min[index] = loss[index]
            x[index, :] = x_new[index, :]

    return x


def expectation_maximization_optimized(
    X: torch.Tensor,
    Y: torch.Tensor,
    A: torch.Tensor,
    steps: int = 3,
    eps: float = 1e-8
):
    """
    Optimized EM algorithm using matrix factorization logic to save memory.
    
    This optimized version reduces memory complexity from O(Batch * M * N) to O(Batch * N)
    by avoiding the creation of large intermediate tensors. It achieves the same mathematical
    result as the standard EM algorithm but with improved computational efficiency.
    
    Memory optimization techniques:
    1. Pre-computing constants outside the loop
    2. Avoiding broadcast operations that create large tensors
    3. Using efficient matrix operations
    
    Args:
        X (torch.Tensor): Input tensor of shape (batch_size, n_features), serves as initial solution
        Y (torch.Tensor): Observed data tensor of shape (batch_size, m_measurements)
        A (torch.Tensor): Transformation matrix of shape (m_measurements, n_features)
        steps (int): Number of iteration steps, default is 3
        eps (float): Numerical stability constant to prevent division by zero, default is 1e-8
        
    Returns:
        torch.Tensor: Optimized X
    """
    # 1. Pre-compute constants for efficiency
    # A_norm: shape (N,) - corresponds to A.sum(dim=0) in the original code
    # Pre-computing avoids repeated calculations in the loop
    A_norm = A.sum(dim=0).clamp_min(eps)
    
    # 2. Initialize variables
    batch_size = X.shape[0]
    # Use X @ A.T instead of (A @ X.T).T, which is more aligned with PyTorch's linear layer conventions and typically faster
    # prediction shape: (Batch, M)
    prediction = X @ A.T 
    
    # Calculate initial loss
    # sum(dim=-1) produces shape (Batch,)
    current_loss = torch.abs(prediction - Y).sum(dim=-1)
    
    x = X.clone()
    
    for _ in range(steps):
        # --- Core optimization section begins ---
        
        # 3. Calculate ratio
        # Corresponds to 1 / linear_sketch(...) * Y in the original code
        # Shape: (Batch, M)
        # Avoids broadcasting to create huge (Batch, M, N) tensors
        ratio = Y / prediction.clamp_min(eps)
        
        # 4. Back-projection
        # Corresponds to einsum('Bb,Bba->Ba', Y, coeff1) in the original code
        # Mathematical essence: Ratio @ A
        # Shape: (Batch, M) @ (M, N) -> (Batch, N)
        grad_factor = ratio @ A
        
        # 5. Update x
        # x_new = x * (grad_factor / A_norm)
        # This is an element-wise operation with minimal memory footprint
        update_factor = grad_factor / A_norm.unsqueeze(0)
        x_new = x * update_factor
        
        # --- Core optimization section ends ---

        # 6. Calculate new loss and update
        new_prediction = x_new @ A.T
        new_loss = torch.abs(new_prediction - Y).sum(dim=-1)
        
        # 7. Conditional update (Vectorized)
        # Using torch.where instead of indexing operations to avoid CPU-GPU synchronization 
        # and maintain tensor contiguity
        # mask shape: (Batch, 1)
        mask = (new_loss < current_loss).unsqueeze(-1)
        
        # Only update samples with improved loss
        x = torch.where(mask, x_new, x)
        
        # Update loss records (note that loss is batch-level)
        current_loss = torch.where(mask.squeeze(-1), new_loss, current_loss)
        
        # Reuse prediction for next iteration (for updated samples)
        # This step is important to avoid recomputing prediction at the beginning of the next loop iteration
        prediction = torch.where(mask, new_prediction, prediction)

    return x
