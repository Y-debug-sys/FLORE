import os
import time 
import torch
import numpy as np

# from tqdm.auto import tqdm
from torch.optim import Adam
from Utils.log import setup_logger
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import MultiStepLR
from Utils.useful_functions import has_nan_or_not
from FLORE.em_opt import expectation_maximization, expectation_maximization_optimized


class SketchingSolver(object):
    """
    A solver class for training flow-based neural networks in the FLORE framework.
    
    This class implements a sketching solver that trains neural networks with
    multiple loss functions including consistency, reconstruction, orthogonal, 
    and invertibility losses. It uses Expectation-Maximization optimization
    and supports various loss functions like Maximum Mean Discrepancy (MMD) 
    and Mean Squared Error (MSE).
    
    The solver manages the complete training pipeline including:
    - Model initialization and device placement
    - Optimizer and learning rate scheduling
    - Loss computation and backpropagation
    - Checkpoint saving and loading
    - Training progress logging
    
    Attributes:
        device (torch.device): The device (CPU/GPU) on which the model runs
        model (torch.nn.Module): The neural network model being trained
        optimizer (torch.optim.Optimizer): The optimizer used for training
        logger (logging.Logger): Logger instance for recording training information
        checkpoint (str): Directory path for saving model checkpoints
        loss_d (callable): Distribution loss function (default: MMD)
        loss_r (callable): Reconstruction loss function (default: MSE)
        alphas (list): Weights for combining different loss terms
    """
    # Predefined loss function identifiers
    MMD = 0
    MSE = 1
    CONSISTENCY = 2
    ORTHOGONALITY = 3
    INVERTIBILITY = 4
    
    def __init__(
        self, 
        args, 
        model, 
        ckpt_name = None,
        optimizer = None, 
        logger = None,
        device = None,
        loss_type: list = None,
        alphas: list = None
    ):
        """
        Initialize the SketchingSolver.
        
        Args:
            args: Configuration arguments containing training parameters
            model (nn.Module): The neural network model to train
            ckpt_name (str, optional): Checkpoint directory name
            optimizer (torch.optim.Optimizer, optional): Custom optimizer
            logger (logging.Logger, optional): Custom logger instance
            device (torch.device, optional): Computing device
            loss_type (list, optional): List of loss function types to use
            alphas (list, optional): Weights for combining different loss terms
        """
        super().__init__()
        # Set computing device (GPU if available, otherwise CPU)
        if device is None: 
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        # Extract training parameters from arguments
        self.num_epochs, self.num_em_steps = args.num_epochs, args.num_em_steps 
        # Move model to designated device
        self.model = model.to(self.device)
        
        # Setup logger if not provided
        if logger is None: 
            logger = setup_logger()
            
        # Setup optimizer if not provided
        if optimizer is None: 
            optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                             lr=args.learning_rate)
            
        self.logger, self.optimizer = logger, optimizer
        # Setup learning rate scheduler with milestones
        mst1, mst2 = args.num_epochs//2, args.num_epochs - args.num_epochs//5
        self.scheduler = MultiStepLR(optimizer, milestones=[mst1, mst2], gamma=0.3)
        
        # Setup checkpoint directory
        self.checkpoint = args.ckpt_dir if ckpt_name is None else ckpt_name
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint, exist_ok=True)

        # Set loss functions based on provided types or defaults
        if loss_type is None:
            self.loss_d = self.MMD
            self.loss_r = self.MSE
        else:
            self.loss_d = loss_type[0]
            self.loss_r = loss_type[1]

        # Set alpha weights for combining different loss terms
        if alphas is None:
            self.alphas = [1., 1., 1., 1., 0.1]
        else:
            if len(alphas) < 5: 
                alphas += [1.] * (5 - len(alphas))
            self.alphas = alphas
            # assert len(alphas) >= 4, "At least four trade-off weights are required."

        if args.print_model:
            # Log model information and parameter count
            self.logger.info(f"Model information: {self.model}\n Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def save(self, filename: str = "model.pt"):
        """
        Save model parameters to a checkpoint file.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        self.logger.info(f"Save trained params to {self.checkpoint + filename}") 
        torch.save(self.model.state_dict(), self.checkpoint + filename) 

    def load(self, filename: str = "model.pt"):
        """
        Load model parameters from a checkpoint file.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        self.logger.info(f"Load pretrained params to {self.checkpoint + filename}") 
        self.model.load_state_dict(data = torch.load(self.checkpoint + filename, map_location=self.device)) 

    def train(self, train_loader, info='round', num_epochs=None):
        """
        Main training function. This function optimizes model parameters through multiple loss terms,
        uses the Expectation-Maximization algorithm for data reconstruction, and logs various metrics 
        during the training process.

        Args:
            train_loader (DataLoader): Data loader providing batches of training samples.
            info (str, optional): Description of the current training phase, defaults to 'round'.
            num_epochs (int, optional): Number of training epochs. If not specified, it will use 
                                       the default value from self.args.num_epochs.

        Returns:
            None: This function does not return a value but updates model parameters and outputs logs.
        """

        # Extract the first four alpha weight coefficients as weights for different loss terms
        w_1, w_2, w_3, w_4, w_5 = self.alphas[:5]

        # Use the default number of epochs from configuration if not specified
        if num_epochs is None:
            num_epochs = self.num_epochs

        # Log the start of training
        self.logger.info(f"Training started.")

        # Start timing
        tic = time.time()
        self.model.train()

        # Calculate index range based on shared dimension (used for expanding input dimensions)
        index_range = self.ndim_x // self.model.share_dim + 1

        try:
            # Enter the training loop
            for epoch in range(1, num_epochs + 1):
                # Log initialization message for current epoch
                self.logger.debug(f"Epoch {epoch}/{num_epochs}: initializing ...")

                # Timestamp at the beginning of this epoch
                etic = time.time()

                # Initialize five loss lists: corresponding to four specific losses and total loss
                losses = [[] for _ in range(6)]

                # Iterate through training batches
                for _, batch_y in enumerate(train_loader):
                    # Get batch size and move data to device
                    batch_size = batch_y.shape[0]
                    batch_y = batch_y.float().to(self.device)
                    batch_y = self.normalize(batch_y)

                    # Expand y dimensions to match model processing requirements
                    y = batch_y.unsqueeze(1).repeat(1, index_range, 1).detach()

                    # Generate random noise vector z
                    z = torch.randn(batch_size, index_range, self.model.ndim_z, device=self.device).float()
                    # print(batch_y.shape, y.shape, z.shape)

                    # Use reverse sampling to get estimated x values hat_x
                    hat_x, hid_y = self.model.reverse_sample(y, z, self.ndim_x)

                    # Run Expectation-Maximization algorithm without gradient tracking to obtain 
                    # more accurate estimate of x values
                    with torch.no_grad():
                        # if epoch < 10:
                            cm_x = self.sketch_min_inverse(batch_y, self.A)
                            x = expectation_maximization_optimized(cm_x, batch_y, self.A, steps=self.num_em_steps).detach()
  
                    # Apply linear transformation A to hat_x to get reconstructed y values rec_y
                    rec_y = (self.A @ x.transpose(0, 1)).transpose(0, 1).unsqueeze(1).repeat(1, index_range, 1)

                    # Input hat_x into model to obtain corresponding y and z representations
                    hat_y, hat_z, hat_hid_y = self.model(hat_x, self.ndim_x)

                    # Perform reverse sampling again to verify invertibility
                    hat_hat_x, _ = self.model.reverse_sample(hat_y, hat_z, self.ndim_x)

                    # Generate new random noise vector z
                    new_z = torch.randn(batch_size, index_range, self.model.ndim_z, device=self.device).float()

                    # Construct new x representation rec_x using reconstructed rec_y and original z
                    rec_x, _ = self.model.reverse_sample(rec_y, new_z, self.ndim_x)

                    # Define various loss terms
                    loss_1 = self.loss_r(hat_y, y)               # Consistency loss
                    loss_2 = self.loss_r(rec_x, x)               # Reconstruction loss
                    loss_3 = self.loss_d(
                        torch.concat([hat_hid_y, hat_z], dim=-1).reshape(batch_size, -1),
                        torch.concat([hid_y, z], dim=-1).reshape(batch_size, -1)
                    )                                            # Orthogonal (or distribution) difference loss
                    loss_4 = self.loss_r(hat_hat_x, hat_x.detach())  # Invertibility loss
                    loss_5 = torch.mean(hat_hat_x)

                    # Weighted sum to get final loss
                    loss = w_1 * loss_1 + w_2 * loss_2 + w_3 * loss_3 + w_4 * loss_4 + w_5 * loss_5

                    # Clear optimizer gradients, backpropagate, clip gradient norm, and perform optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()

                    # Store results of each loss term for subsequent statistical analysis
                    losses[0].append(loss_1.item())
                    losses[1].append(loss_2.item())
                    losses[2].append(loss_3.item())
                    losses[3].append(loss_4.item())
                    losses[4].append(loss_5.item())
                    losses[5].append(loss.item())

                # Calculate average values of each loss type in this epoch
                avg_losses = [np.average(loss_list) for loss_list in losses]

                # Build log message string
                log_msg = (
                    f"Epoch {epoch} done. "
                    f"|| consistency loss: {avg_losses[0]:.2f}; "
                    f"reconstruction loss: {avg_losses[1]:.2f}; "
                    f"orthogonal loss: {avg_losses[2]:.2f}; "
                    f"invertible loss: {avg_losses[3]:.2f}; "
                    f"sparse loss: {avg_losses[4]:.2f} | "
                    f"|| average loss: {avg_losses[5]:.4f}; "
                    f"cost time: {(time.time() - etic):.4f} s."
                )

                # Output completion information and performance metrics for this round of training
                self.logger.info(log_msg)
                self.scheduler.step()

        except Exception as e:
            # Log error and re-raise exception when an error occurs
            self.logger.error(f"Error occurred during training: {str(e)}", exc_info=True)
            raise

        # Overall training time statistics and completion notification
        total_time_min = (time.time() - tic) / 60
        self.logger.info(f"Training completed. Total cost time: {total_time_min:.2f} mins.")

        # Save model
        self.save(filename=f"model_{info}.pt")

    def update_sys(self, ndim_x: int, mat: csr_matrix) -> None:
        """ Update the sketching operator and dimensionality (length) of the frequency vector."""

        if not isinstance(ndim_x, int) or ndim_x <= 0:
            self.logger.error(f"Invalid ndim_x: {ndim_x}. It must be a positive integer.")
            raise ValueError(f"Invalid ndim_x: {ndim_x}. Must be positive integer.")

        old_value = getattr(self, "ndim_x", None)
        self.ndim_x = ndim_x
        self.logger.debug(f"Vector dimensionality updated from {old_value} to {ndim_x}.")

        self.A = torch.from_numpy(mat).float().to(self.device)

    @torch.no_grad()
    def test_in_batch(self, test_loader):
        """
        Test the trained model on given data loader and return the results in batch.
        
        This method evaluates the model in inference mode without computing gradients,
        generates samples using the reverse sampling technique, and returns the final results.
        
        Args:
            test_loader: DataLoader containing test data batches
            
        Returns:
            numpy.ndarray: Generated samples as numpy array on CPU
        """
        results = []
        self.model.eval()
        t_start = time.time()
        index_range = self.ndim_x // self.model.share_dim + 1

        try:
            for _, batch_y in enumerate(test_loader):
                batch_size = batch_y.shape[0]
                batch_y = batch_y.float().to(self.device)
                batch_y, scale = self.normalize(batch_y, return_scale=True)

                y = batch_y.unsqueeze(1).repeat(1, index_range, 1).detach()
                z = torch.randn(batch_size, index_range, self.model.ndim_z, device=self.device).float()
                x = self.denormalize(self.model.reverse_sample(y, z, self.ndim_x)[0], scale)
                results.append(x)
            
            results = torch.cat(results, dim=0)
            self.model.train()

            return results.detach().cpu().numpy()

        except KeyboardInterrupt: pass
        finally:  print(f"\n\nTesting took {(time.time() - t_start):.2f} s\n")

    @torch.no_grad()
    def test_in_sample(self, counters: np.ndarray):
        """
        Test on a single sample and generate reverse sampling results.
        
        This method evaluates the model in inference mode without computing gradients,
        generates samples using the given counter data, and returns the final results.
        
        Args:
            counters (np.ndarray): Input counter data used to generate corresponding samples
            
        Returns:
            numpy.ndarray: Generated sample data converted to numpy array on CPU
        """
        self.model.eval()
        index_range = self.ndim_x // self.model.share_dim + 1

        batch_y = torch.from_numpy(counters).float().unsqueeze(0).to(self.device)
        batch_y, scale = self.normalize(batch_y, return_scale=True)

        y = batch_y.unsqueeze(1).repeat(1, index_range, 1).detach()
        z = torch.randn(1, index_range, self.model.ndim_z, device=self.device).float()
        x = self.denormalize(self.model.reverse_sample(y, z, self.ndim_x)[0], scale)
        
        self.model.train()
        return x.squeeze(0).detach().cpu().numpy()

    @staticmethod
    def MMD(x: torch.Tensor, y: torch.Tensor, scales: list = [0.05, 0.2, 0.9]):
        """
        Calculate the Maximum Mean Discrepancy (MMD) distance between two sets of samples
        
        This function uses multi-scale RBF kernels to compute the MMD distance, which measures 
        the discrepancy between two distributions.
        
        Parameters:
            x (torch.Tensor): First sample set with shape (n1, d), where n1 is the number of samples 
                              and d is the feature dimension
            y (torch.Tensor): Second sample set with shape (n2, d), where n2 is the number of samples 
                              and d is the feature dimension
            scales (list): List of scale parameters for the RBF kernel, default is [0.05, 0.2, 0.9]
        
        Returns:
            torch.Tensor: The MMD distance value between the two sample sets
        """
        device = x.device

        # Compute inner product matrices for within-sample and between-sample comparisons
        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        # Calculate diagonal matrices of squared distances for each sample to itself, 
        # then expand to match the shape of dot product matrices
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        # Compute squared Euclidean distance matrices: dij = ||xi||^2 + ||xj||^2 - 2*<xi,xj>
        dxx = rx.t() + rx - 2.*xx
        dyy = ry.t() + ry - 2.*yy
        dxy = rx.t() + ry - 2.*zz

        # Initialize kernel matrices
        XX, YY, XY = (torch.zeros(xx.shape, device=device),
                      torch.zeros(xx.shape, device=device),
                      torch.zeros(xx.shape, device=device))

        # Compute MMD distance using multi-scale RBF kernels
        for a in scales:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

        # Return the mean MMD distance
        return torch.mean(XX + YY - 2.*XY)
    
    @staticmethod
    def MAE(x1: torch.Tensor, x2: torch.Tensor):
        """
        Calculate the Mean Absolute Error (MAE) between two tensors
        
        Args:
            x1 (torch.Tensor): First input tensor
            x2 (torch.Tensor): Second input tensor
            
        Returns:
            torch.Tensor: Mean absolute error value between the two tensors
        """
        return torch.mean((x1 - x2).abs())
    
    @staticmethod
    def MSE(x1: torch.Tensor, x2: torch.Tensor):
        """
        Calculate the Mean Squared Error (MSE) between two tensors
        
        Args:
            x1 (torch.Tensor): First input tensor
            x2 (torch.Tensor): Second input tensor
            
        Returns:
            torch.Tensor: Mean squared error value between the two tensors
        """
        return torch.mean((x1 - x2)**2)

    @staticmethod
    def KLD(p: torch.Tensor, q: torch.Tensor):
        """
        Computes the Kullback-Leibler Divergence between two distributions.
    
        Formula: KL(P || Q) = sum(P * log(P / Q))
    
        Args:
            p (torch.Tensor): The target probability distribution (ground truth).
                              Shape: [batch_size, feature_size]
            q (torch.Tensor): The predicted probability distribution.
                              Shape: [batch_size, feature_size]
                          
        Returns:
            torch.Tensor: A scalar tensor representing the average KL divergence.
        """
    
        # Small constant to prevent numerical instability (log(0) resulting in NaN)
        epsilon = 1e-10
    
        # Ensure inputs are treated as probabilities (clamping for safety)
        # This prevents exact 0 or 1 which can cause math errors in log
        p = torch.clamp(p, min=epsilon, max=1.0)
        q = torch.clamp(q, min=epsilon, max=1.0)
    
        # Calculate the point-wise divergence
        # KL = p * (log(p) - log(q)) is mathematically equivalent to p * log(p/q)
        divergence_pointwise = p * (torch.log(p) - torch.log(q))
    
        # Sum over the feature dimension (dim=1) to get KL per sample
        # Shape becomes: [batch_size]
        kl_per_sample = torch.sum(divergence_pointwise, dim=1)

        return torch.mean(kl_per_sample)
    
    @staticmethod
    def normalize(x, scale=None, return_scale=False):
        """ Instance normalization """
        batch_size = x.shape[0]
        scale = torch.min(torch.max(x, dim=-1)[0], dim=-1)[0] if scale is None else scale
        scale = torch.max(scale, torch.tensor(1e-5))
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        if not return_scale: return (x / scale).reshape(batch_size, -1)
        return (x / scale).reshape(batch_size, -1), scale.squeeze(-1)

    @staticmethod
    def denormalize(x, scale):
        """ Instance denormalization """
        x = x * scale
        return torch.ceil(x)
    
    @staticmethod
    def sketch_min_inverse(Y: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs features X from measurements Y and sketch matrix A using the 'Min' rule.
        (Commonly used in Count-Min Sketch retrieval).
        
        Logic: For each feature column j in A, find all rows i where A[i, j] == 1.
               Then X[b, j] = min(Y[b, i]) for all such i.
               
        Parameters:
            Y (torch.Tensor): Observed measurements (Batch, M_measurements)
            A (torch.Tensor): Binary sketch matrix (M_measurements, N_features)
                              Elements should be 0 or 1.
                              
        Returns:
            torch.Tensor: Estimated X (Batch, N_features)
        """
        
        # 1. Determine the maximum number of hash functions (1s) per column.
        #    For a standard CM-Sketch, this is constant (e.g., d=4).
        #    We calculate it dynamically to handle potentially irregular matrices.
        #    Shape: scalar
        k = int(A[:, 0].sum().item())
        
        # 2. Extract indices of the '1's in A.
        #    We use topk to efficiently find the row indices 'i' for each column 'j'.
        #    'indices' will contain the row index (0 to M-1).
        #    'values' will verify if it's a true 1 or a padding 0 (if columns have varying 1s).
        #    Shape: (k, N_features)
        values, indices = torch.topk(A, k=k, dim=0)
        
        # 3. Gather values from Y.
        #    PyTorch Advanced Indexing allows us to index the second dimension of Y (dimension 1)
        #    using the 'indices' tensor directly.
        #    Y: (Batch, M)
        #    indices: (k, N)
        #    Result 'gathered': (Batch, k, N)
        gathered = Y[:, indices]
    
        # 4. Compute the Minimum.
        #    Reduce along the 'k' dimension (dimension 1).
        #    Shape: (Batch, N_features)
        x_estimated = gathered.min(dim=1).values
        
        return x_estimated


if __name__ == '__main__':
    pass
