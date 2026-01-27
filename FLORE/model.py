import torch
# import warnings

from torch import nn
from torch import Tensor
from Flows.layers import GLOWCouplingBlock
from typing import Union, Iterable, Tuple
from Flows.nodes import Node, ConditionNode, InputNode, OutputNode
from Flows.inns import GraphINN, PermuteRandom, InvertibleSigmoid, InvertibleTanh


def one_hot(labels, dim):
    '''
    Convert LongTensor labels (contains labels 0-X), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    
    Args:
        labels: A tensor of label indices to convert to one-hot encoding
        dim: The dimension of the one-hot encoded output
        
    Returns:
        Tensor: One-hot encoded tensor of shape (labels.shape[0], dim)
    '''
    # Create a zero tensor with shape (batch_size, dim)
    out = torch.zeros(labels.shape[0], dim).to(labels.device)
    # Scatter 1.0 values at the specified indices to create one-hot vectors
    out.scatter_(dim=1, index=labels.view(-1,1), value=1.)
    return out


class AutoEncoder(nn.Module):
    """
    AutoEncoder for dimensionality reduction and feature compression in the FLORE framework.
    
    This autoencoder consists of an encoder that compresses input data to a lower-dimensional
    representation and a decoder that reconstructs the original data from the compressed form.
    It is used in the FLORE framework to process high-dimensional sketch data efficiently.
    
    The architecture includes:
    - Encoder: Linear -> ReLU -> Dropout -> Linear -> Tanh
    - Decoder: Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    
    The encoder maps the input to a compressed representation in a tanh-activated space,
    while the decoder reconstructs the input in a sigmoid-activated space (suitable for 
    normalized data in the [0, 1] range).
    
    Attributes:
        encoder (nn.Sequential): The encoder network that compresses input data
        decoder (nn.Sequential): The decoder network that reconstructs data from compressed representation
    """
    
    def __init__(self, in_dim, compress_dim, dp_ratio=0.1, **kwargs):
        """
        Initialize the AutoEncoder with encoder and decoder networks.
        
        Args:
            in_dim (int): Dimension of the input data to be encoded/decoded
            compress_dim (int): Dimension of the compressed representation (bottleneck layer)
            dp_ratio (float): Dropout ratio for regularization, defaults to 0.1
            **kwargs: Additional arguments passed to the parent nn.Module class
        """
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, compress_dim),
            nn.ReLU(),
            nn.Dropout(dp_ratio),
            nn.Linear(compress_dim, compress_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(compress_dim, compress_dim),
            nn.ReLU(),
            nn.Dropout(dp_ratio),
            nn.Linear(compress_dim, in_dim),
            nn.Sigmoid()
        )

    def encode(self, input):
        """
        Encode input data to compressed representation.
        
        Passes the input through the encoder network to produce a lower-dimensional
        representation that captures the essential features of the input data.
        
        Args:
            input (Tensor): Input data tensor of shape (batch_size, in_dim)
            
        Returns:
            Tensor: Compressed representation of shape (batch_size, compress_dim)
        """
        return self.encoder(input)

    def decode(self, input):
        """
        Decode compressed representation back to original space.
        
        Passes the compressed representation through the decoder network to 
        reconstruct the original input data as closely as possible.
        
        Args:
            input (Tensor): Compressed representation of shape (batch_size, compress_dim)
            
        Returns:
            Tensor: Reconstructed data of shape (batch_size, in_dim)
        """
        return self.decoder(input)


class ReversibleGraphNet(GraphINN):
    '''
    A reversible graph network that extends GraphINN with a custom forward method.
    '''
    def __init__(
        self, 
        node_list: list, 
        verbose: bool = True, 
        force_tuple_output: bool = False 
    ): 
        '''
        Initialize the ReversibleGraphNet.
        
        Args:
            node_list: List of nodes defining the network architecture
            verbose: Whether to print detailed information during execution
            force_tuple_output: Whether to force tuple output
        '''
        super().__init__(node_list, verbose=verbose,
                         force_tuple_output=force_tuple_output)

    def forward(self, x_or_z: Union[Tensor, Iterable[Tensor]],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True,
                intermediate_outputs: bool = False) -> Tuple[Tuple[Tensor], Tensor]:
        '''
        Forward pass through the network.
        
        Args:
            x_or_z: Input tensor(s) to the network
            c: Conditional input tensors
            rev: Whether to run in reverse mode
            jac: Whether to compute Jacobian
            intermediate_outputs: Whether to return intermediate outputs
            
        Returns:
            Tuple of (output tensors, jacobian)
        '''
        return super().forward(x_or_z, c, rev, jac, intermediate_outputs)


class FLORE_cINN(nn.Module):
    '''INN for index-conditional estimation problem'''
    def __init__(
        self, 
        x_dim: int = 2048, 
        y_dim: int = 1024, 
        num_layers: int = 8, 
        hidden_dim: int = 512,
        max_cond_dim: int = 100,
        verbose: bool = False
    ):
        '''
        Initialize the FLORE conditional INN (Invertible Neural Network).
        
        Args:
            x_dim: Shared dimension for the input/output
            y_dim: Number of counter variables
            num_layers: Number of coupling layers in the network
            hidden_dim: Hidden dimension for subnet construction
            max_cond_dim: Maximum dimension for conditional input
            verbose: Whether to print detailed information during execution
        '''
        super().__init__()

        def subnet_fc(ch_in, ch_out):
            '''
            Factory function to create fully connected subnet blocks.
            
            Args:
                ch_in: Input channel dimension
                ch_out: Output channel dimension
                
            Returns:
                Sequential neural network block
            '''
            return nn.Sequential(nn.Linear(ch_in, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, ch_out))
                                 
        self.ndim_y = y_dim
        self.ndim_z = x_dim - y_dim
        self.share_dim = x_dim

        # Ensure the shared dimension is larger than the number of counters
        if self.ndim_z <= 0:
            raise ValueError(f"Shared output size {x_dim} must >= \
                               the number of counters {y_dim}.")

        self.max_cond_dim = max_cond_dim
        # Create condition node for conditional input
        cond = ConditionNode(max_cond_dim)
        # Start with input node
        nodes = [InputNode(x_dim, name='input')]

        # Construct coupling layers with permutations
        for k in range(num_layers):
            # Add coupling block with subnet constructor
            nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                              {'subnet_constructor':subnet_fc, 'clamp':2.0},
                              name=f'coupling_{k}', conditions=cond))
            # Add permutation or sigmoid layer depending on position
            if k < num_layers - 1:
                nodes.append(Node(nodes[-1], PermuteRandom, {'seed':k}, name=f'permute_{k}'))
            else:
                nodes.append(Node(nodes[-1], InvertibleTanh, name=f'tanh'))

        # Create the final network with all nodes
        self.cinn = ReversibleGraphNet(nodes + [cond, OutputNode(nodes[-1], name='output')], verbose=verbose)
    
    def forward(self, x, index_range, batch_size, jac=False):
        # Create index tensor for conditional input
        index_instance = torch.arange(index_range, device=x.device, dtype=torch.long).unsqueeze(0)
        index_tensor = index_instance.repeat(batch_size, 1).reshape(-1,)
        # Pass through the network with one-hot encoded indices
        output = self.cinn(x, c=one_hot(index_tensor, self.max_cond_dim), jac=jac)[0]

        # Reshape outputs to batch format
        y, z = output[:, :self.ndim_y].reshape(batch_size, -1, self.ndim_y), \
               output[:, -self.ndim_z:].reshape(batch_size, -1, self.ndim_z)
        
        # y, z = torch.mean(y, dim=1), torch.mean(z, dim=1) 

        if jac:
            # Compute and return Jacobian if requested
            jac = self.cinn.log_jacobian(run_forward=False)
            return z, y, jac
        else:
            return z, y

    def reverse_sample(self, y, z, index_range, jac=False):
        # Concatenate y and z tensors
        input = torch.cat((y.reshape(-1, y.shape[-1]), z.reshape(-1, z.shape[-1])), dim=1)
        # Create index tensor for conditional input
        index_instance = torch.arange(index_range, device=y.device, dtype=torch.long).unsqueeze(0)
        index_tensor = index_instance.repeat(y.shape[0], 1).reshape(-1,)
        # Run the network in reverse mode
        output = self.cinn(input, c=one_hot(index_tensor, self.max_cond_dim), rev=True, jac=jac)[0]
        return output


class FLOREModel(nn.Module):
    """
    Main FLORE model integrating autoencoders with conditional invertible neural networks.
    
    This model serves as the core generative component of the FLORE framework, combining
    autoencoding techniques with conditional invertible neural networks (cINNs) to learn
    and reconstruct data distributions from sketch measurements.
    
    The architecture consists of three main components:
    1. Counter AutoEncoder: Compresses and reconstructs sketch counter data (Y-space)
    2. Vector AutoEncoder: Compresses and reconstructs original data vectors (X-space)
    3. FLORE_cINN: Conditional invertible neural network modeling the relationship between
       compressed X and Y representations
    
    The model operates in both forward and reverse modes:
    - Forward: Maps data vectors to reconstructed sketch counters
    - Reverse: Samples data vectors from sketch counter observations
    
    Attributes:
        counter_ae (AutoEncoder): AutoEncoder for sketch counter data (Y-space)
        vector_ae (AutoEncoder): AutoEncoder for original data vectors (X-space)
        cinn (FLORE_cINN): Conditional invertible neural network modeling X-Y relationships
        share_dim (int): Shared dimension for data processing
        ndim_z (int): Dimension of latent space in the cINN
    """
    
    def __init__(
        self, 
        x_dim: int, 
        xh_dim: int, 
        y_dim: int, 
        yh_dim: int, 
        dropout: float = 0., 
        num_layers: int = 8, 
        hidden_dim: int = 512,
        max_cond_dim: int = 100,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the FLOREModel with all its components.
        
        Args:
            x_dim (int): Dimension of the original data vectors (input space)
            xh_dim (int): Dimension of compressed representation of data vectors
            y_dim (int): Dimension of sketch counter data (measurement space)
            yh_dim (int): Dimension of compressed representation of counter data
            dropout (float): Dropout ratio for regularization in autoencoders, defaults to 0.
            num_layers (int): Number of coupling layers in the cINN, defaults to 8
            hidden_dim (int): Hidden dimension for neural network layers, defaults to 512
            max_cond_dim (int): Maximum condition dimension for the cINN, defaults to 100
            verbose (bool): Whether to print detailed information during execution, defaults to False
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Initialize autoencoder for sketch counter data (Y-space compression/reconstruction)
        self.counter_ae = AutoEncoder(y_dim, yh_dim, dropout)
        # Initialize autoencoder for original data vectors (X-space compression/reconstruction)
        self.vector_ae = AutoEncoder(x_dim, xh_dim, dropout)
        # Initialize conditional invertible neural network modeling the X-Y relationship
        self.cinn = FLORE_cINN(xh_dim, yh_dim, num_layers, hidden_dim, max_cond_dim, verbose)
        # Store shared dimension for data processing
        self.share_dim = x_dim
        # Store latent space dimension from the cINN
        self.ndim_z = self.cinn.ndim_z

    def forward(self, x, ndim_x):
        """
        Forward pass through the FLORE model.
        
        Processes input data vectors through the full pipeline:
        1. Pads inputs to match required dimensions
        2. Encodes data vectors using the vector autoencoder
        3. Passes encoded vectors through the cINN to generate latent representations and predicted counters
        4. Decodes predicted counters using the counter autoencoder
        
        Args:
            x (Tensor): Input data vectors of shape (batch_size, ndim_x)
            ndim_x (int): Actual dimension of input data (may be less than padded dimension)
            
        Returns:
            tuple: (y_dec, z, y) where:
                - y_dec: Reconstructed counter data of shape (batch_size, y_dim)
                - z: Latent representations from the cINN
                - y: Compressed counter representations before decoding
        """
        batch_size = x.shape[0]
        index_range = ndim_x // self.share_dim + 1
        # Pad input to match share dimension for consistent processing
        pad_x = torch.zeros(x.shape[0], self.share_dim - ndim_x % self.share_dim, device=x.device)
        x = torch.cat((x, pad_x), dim=1).reshape(-1, self.share_dim)
        # Encode input data vectors to compressed representation
        x_enc = self.vector_ae.encode(x)
        # Pass through cINN to generate latent representations and counter predictions
        z, y = self.cinn(x_enc, index_range, batch_size)
        # Decode predicted counters to original space
        y_dec = self.counter_ae.decode(y)
        return y_dec, z, y

    def reverse_sample(self, y, z, ndim_x):
        """
        Reverse sampling pass through the FLORE model.
        
        Generates data vectors from observed counter data by reversing the process:
        1. Encodes counter data using the counter autoencoder
        2. Uses the cINN in reverse mode to sample data vectors from counter representations and latent variables
        3. Decodes sampled vectors using the vector autoencoder
        4. Reshapes and trims output to match original data dimensions
        
        Args:
            y (Tensor): Observed counter data of shape (batch_size, y_dim)
            z (Tensor): Latent variables for sampling of shape (batch_size, ndim_z)
            ndim_x (int): Target dimension for output data vectors
            
        Returns:
            tuple: (x_dec, y_enc) where:
                - x_dec: Sampled and decoded data vectors of shape (batch_size, ndim_x)
                - y_enc: Encoded counter representations
        """
        batch_size = y.shape[0]
        index_range = ndim_x // self.share_dim + 1
        # Encode counter data to compressed representation
        y_enc = self.counter_ae.encode(y)
        # Reverse sample data vectors from counter representations and latent variables
        x = self.cinn.reverse_sample(y_enc, z, index_range)
        # Decode sampled vectors to original space
        x_dec = self.vector_ae.decode(x)
        # Reshape and trim output to match original data dimensions
        return x_dec.reshape(batch_size, -1)[:, :ndim_x], y_enc
