# ======================================================================================
# Source / Reference:
#   - Chen-Yu Hsu
#   - https://github.com/chenyuhsu/learnedsketch/blob/master/run_aol_model.py
#
# License:
#   MIT License
#   https://github.com/chenyuhsu/learnedsketch/blob/master/LICENSE
# ======================================================================================


import torch

from torch import nn


class OracleModel(nn.Module):
    """
    A neural network model that combines embedding, RNN, and fully connected layers
    for processing sequential data and predicting numerical outputs.
    
    This model processes variable-length sequences by:
    1. Embedding discrete tokens into continuous vector space
    2. Processing sequences with RNN (LSTM/GRU) to capture temporal dependencies
    3. Using final hidden state + sequence length as features for fully connected layers
    4. Producing a scalar prediction
    
    Args:
        embed_size (int): Size of the embedding vectors for each character/token. Default: 128
        cell_type (str): Type of RNN cell to use ('LSTM' or 'GRU'). Default: "LSTM"
        rnn_hiddens (list): List of hidden dimensions for RNN layers. Default: [128]
        hiddens (list): List of hidden dimensions for fully connected layers. Default: [128]
        activation (str): Activation function to use ('relu' or 'tanh'). Default: "relu"
        dropout_rate (float): Dropout rate for regularization in fully connected layers. Default: 0.
    """
    def __init__(
        self, 
        embed_size=16, 
        cell_type="LSTM", 
        rnn_hiddens=[16], 
        hiddens=[16], 
        activation="relu", 
        dropout_rate=0.
    ):
        super(OracleModel, self).__init__()

        # Model hyperparameters
        self.embed_size = embed_size
        self.alphabet_size = 44  # Size of vocabulary/character set
        self.cell_type = cell_type
        self.rnn_hiddens = rnn_hiddens
        self.activation = activation
        # self.relu_output = True

        # Create embedding layer for converting characters to dense vectors
        self.embedding = nn.Embedding(self.alphabet_size, self.embed_size)

        # Initialize RNN layers based on specified cell type
        if cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.embed_size,
                hidden_size=self.rnn_hiddens[0],
                num_layers=len(self.rnn_hiddens),
                batch_first=True
            )
        elif cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.embed_size,
                hidden_size=self.rnn_hiddens[0],
                num_layers=len(self.rnn_hiddens),
                batch_first=True
            )
        else:
            raise ValueError("Unknown cell type %s. Supported types: LSTM, GRU" % cell_type)

        # Construct fully connected layers for final prediction
        # Input dimension includes sequence length feature plus RNN final hidden state
        input_dim = 1 + self.rnn_hiddens[-1]   # feat_len + final_state
        hidden_dims = hiddens

        # Build layer dimensions from input to output
        layer_sizes = [input_dim] + hidden_dims + [1]

        # Create sequential layers with activations and dropout
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                if self.activation == "relu":
                    layers.append(nn.ReLU())
                elif self.activation == "tanh":
                    layers.append(nn.Tanh())
                else:
                    raise ValueError("Unsupported activation %s. Supported: relu, tanh" % self.activation)
                # dropout for each hidden layer
                layers.append(nn.Dropout(dropout_rate))

        self.fc = nn.Sequential(*layers)

    def forward(self, feat, labels=None, data_len=None):
        """
        Forward pass through the model.
        
        Args:
            feat (Tensor): Input features of shape [B, n_feat] where:
                          feat[:,0] contains sequence lengths
                          feat[:,1:] contains character sequences
            labels (Tensor, optional): Ground truth labels for loss calculation (unused in current implementation)
            data_len (int, optional): Length of valid data for masking (unused in current implementation)
            
        Returns:
            Tensor: Model predictions of shape [B]
        """
        # Separate length + characters from input features
        # Using direct indexing instead of slicing for better performance (based on experience lesson)
        feat_len = feat[:, 0].long()                # [B]
        
        # More efficient way to extract character features without creating intermediate slices
        batch_size = feat.size(0)
        feat_chars = torch.zeros(batch_size, feat.size(1) - 1, dtype=torch.long, device=feat.device)
        for i in range(batch_size):
            feat_chars[i] = feat[i, 1:]             # [B, T]
        
        # Alternative approach for better GPU utilization - vectorized indexing
        # feat_char = feat[:, 1:].long()              # [B, T]

        # Convert characters to embeddings
        embeds = self.embedding(feat_chars)         # [B, T, embed]

        # Pack for variable-length RNN processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds,
            lengths=feat_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        # Process sequence through RNN
        if self.cell_type == "LSTM":
            _, (h_n, _) = self.rnn(packed)
        else:  # GRU
            _, h_n = self.rnn(packed)

        # Extract final hidden state from the last layer
        final_state = h_n[-1]                       # [B, hidden]

        # Concatenate sequence length feature with final RNN state
        feat_mid = torch.cat(
            [feat_len.unsqueeze(1).float(), final_state],
            dim=1
        )                                            # [B, 1+hidden]

        # Pass through fully connected layers for final prediction
        output = self.fc(feat_mid).squeeze(-1)       # [B]

        # if self.relu_output:
        #     output = F.relu(output)

        # # -------------------------------
        # # Match TF behavior: loss on [:data_len]
        # # -------------------------------
        # masked_output = output[:data_len]
        # masked_labels = labels[:data_len]

        # loss = F.mse_loss(masked_output, masked_labels)

        # return output, loss

        return output
    
    def get_memory_usage(self):
        """
        Calculate the total memory usage of the model parameters and buffers.
        
        This function computes the memory footprint by iterating through all model
        parameters and buffers, multiplying the number of elements by the size of
        each element in bytes.
        
        Returns:
            int: Total memory usage in bytes
        """
        total_bytes = 0

        # Count memory used by model parameters (weights and biases)
        for param in self.parameters():
            total_bytes += param.nelement() * param.element_size()

        # Count memory used by model buffers (e.g., running statistics in BatchNorm)
        for buf in self.buffers():
            total_bytes += buf.nelement() * buf.element_size()

        return total_bytes
    
    def run_single_simulation(self, device=None):
        """
        Run a single simulation with randomly generated input data.
        
        This function generates random sequence data and feeds it through the model
        to produce a single prediction output. It's useful for testing and demonstration
        purposes, allowing quick evaluation of the model behavior without external data.
        
        Args:
            device (torch.device, optional): The device (CPU/GPU) to run the simulation on.
                                           If None, uses the current device. Default: None
            
        Returns:
            Tensor: Model prediction output for the generated random input
        """
        # Set batch size and sequence parameters
        B = 1
        T = self.alphabet_size - 1 
        vocab_size = self.embedding.num_embeddings 

        # Generate random sequence lengths and character sequences
        lengths = torch.randint(low=1, high=T+1, size=(B,))
        chars = torch.randint(low=0, high=vocab_size, size=(B, T))
        input = torch.cat([lengths.unsqueeze(1), chars], dim=1)

        # Move input to specified device if provided
        if device is not None:
            input = input.to(device)

        return self.forward(input)
