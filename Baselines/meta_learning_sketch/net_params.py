import torch

from torch import nn


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self,input_size, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()
        self.range = nn.Parameter(torch.arange(start=1, end=input_size + 1, step=1, dtype=torch.float).view(1, -1),requires_grad=False)
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = self.range
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
    

####################
# Attention Matrix #
####################
    

class AttentionMatrix(nn.Module):
    def __init__(self, refined_dim, slot_dim, depth_dim = 1):
        super().__init__()
        self.refined_dim = refined_dim
        self.slot_dim = slot_dim
        self.attention_matrix = torch.nn.Parameter(torch.rand(depth_dim,refined_dim, slot_dim,  requires_grad=True))
        self.normalize()
        self.sparse_softmax = Sparsemax(slot_dim)

    def forward(self, refined_vec):
        product_tensor = refined_vec.matmul(self.attention_matrix)
        return self.sparse_softmax(product_tensor)

    def normalize(self):
        with torch.no_grad():
            matrix_pow_2 = torch.square(self.attention_matrix)
            matrix_base = torch.sqrt(matrix_pow_2.sum(dim=1, keepdim=True))
            # automatic broadcast
            self.attention_matrix.data = self.attention_matrix.div(matrix_base)


####################
# Embedding Module #
####################


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, (self.hidden_layer_size + self.out_dim) // 2),
            nn.BatchNorm1d((self.hidden_layer_size + self.out_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.hidden_layer_size + self.out_dim) // 2, self.out_dim),
            # nn.LayerNorm(self.out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingNetDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = hidden_layer_size
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, (self.hidden_layer_size + self.out_dim) // 2),
            nn.BatchNorm1d((self.hidden_layer_size + self.out_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.hidden_layer_size + self.out_dim) // 2, self.out_dim),
            # nn.LayerNorm(self.out_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.net(x)
    

###################
# Decoding Module #
###################


class WeightDecodeNet(nn.Module):
    def __init__(self, input_dim, weight_decode_hidden_layer_size, output_dim=1):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = weight_decode_hidden_layer_size
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer_size, (self.hidden_layer_size + self.out_dim) // 2),
            nn.BatchNorm1d((self.hidden_layer_size + self.out_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.hidden_layer_size + self.out_dim) // 2, self.out_dim),
        )

    def forward(self, x):
        return self.net(x)


class WeightDecodeNetResidual(nn.Module):
    def __init__(self, input_dim, weight_decode_hidden_layer_size, output_dim=1):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = weight_decode_hidden_layer_size
        self.hidden_1 = nn.Linear(self.in_dim, self.hidden_layer_size)
        self.activate_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(self.hidden_layer_size, self.in_dim)
        self.activate_2 = nn.ReLU()
        self.hidden_3 = nn.Linear(self.in_dim, (self.in_dim + self.out_dim) // 2)
        self.activate_3 = nn.ReLU()
        self.out = nn.Linear((self.in_dim + self.out_dim) // 2, self.out_dim)

    def forward(self, x):
        y = (self.hidden_1(x))
        y = self.activate_1(y)
        y = (self.activate_2((self.hidden_2(y)) + x))
        y = (self.activate_3((self.hidden_3(y))))
        return self.out(y)


class ResExistDecodeNet(nn.Module):
    def __init__(self, input_dim, weight_decode_hidden_layer_size, output_dim=1):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = weight_decode_hidden_layer_size
        self.hidden_1 = nn.Linear(self.in_dim, self.hidden_layer_size)
        self.ln1 = nn.LayerNorm(self.hidden_layer_size)

        self.activate_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(self.hidden_layer_size, self.in_dim)
        self.activate_2 = nn.ReLU()
        self.hidden_3 = nn.Linear(self.in_dim, (self.in_dim + self.out_dim) // 2)
        self.activate_3 = nn.ReLU()
        self.ln3 = nn.LayerNorm((self.in_dim + self.out_dim) // 2)
        self.out = nn.Linear((self.in_dim + self.out_dim) // 2, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.hidden_1(x)
        y = self.activate_1(self.ln1(y))
        y = self.activate_2(self.hidden_2(y) + x)
        y = self.activate_3(self.ln3(self.hidden_3(y)))
        y = self.out(y)

        return self.sigmoid(y)
    

###################
# Refining Module #
###################


class RefineNet(nn.Module):
    def __init__(self, source_embedding_dim, source_refined_dim, source_refined_hidden_layer_size):
        super().__init__()
        self.in_dim = source_embedding_dim
        self.out_dim = source_refined_dim
        self.hidden_layer_size = source_refined_hidden_layer_size
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer_size, (self.hidden_layer_size + self.out_dim) // 2),
            nn.BatchNorm1d((self.hidden_layer_size + self.out_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.hidden_layer_size + self.out_dim) // 2, self.out_dim),
        )

    def forward(self, x):
        return self.net(x)


#################
# Memory Matrix #
#################


class BasicMemoryMatrix(nn.Module):
    def __init__(self, depth_dim, slot_dim, embedding_dim):
        super().__init__()
        self.slot_dim = slot_dim
        self.depth_dim = depth_dim
        self.embedding_dim = embedding_dim
        # self.memory_matrix = nn.Parameter(torch.zeros(depth_dim,embedding_dim,slot_dim, requires_grad=False))
        self.memory_matrix = None
        self.device = None

    def clear(self):
        with torch.autograd.no_grad():
            self.memory_matrix = (torch.zeros(self.depth_dim, self.slot_dim, self.embedding_dim,device=self.device, requires_grad=True))

    # address dim : depth * batch size * slot
    def write(self, address, embedding, frequency):
        frequency_embedding = embedding * frequency.view(-1, 1)
        write_matrix = address.transpose(1, 2).matmul(frequency_embedding)
        self.memory_matrix = self.memory_matrix + write_matrix

    def read(self, address, embedding):
        read_info_tuple = self.basic_read_attention_sum(address, embedding)
        return torch.cat(read_info_tuple, dim=-1)

    # address dim : depth * batch size * slot
    def basic_read_attention_sum(self, address, embedding, read_compensate=True):
        batch_size = address.shape[1]
        basic_read_matrix = address.matmul(self.memory_matrix)
        if read_compensate:
            basic_read_matrix = (1 / address.square().sum(dim=-1, keepdim=True)) * basic_read_matrix
        # basic_read_matrix dim: batch size * depth * embedding
        cm_embedding = torch.where(embedding > 0.00001, embedding, torch.zeros_like(embedding) + 0.00001)
        zero_add_vec = torch.where(abs(embedding) < 0.0001,torch.zeros_like(embedding)+10000,torch.zeros_like(embedding))
        cm_read_info_1 = self.cm_read_1(basic_read_matrix, cm_embedding,zero_add_vec)
        cm_read_info_2 = self.cm_read_2(basic_read_matrix, cm_embedding,zero_add_vec)
        basic_read_info = basic_read_matrix.transpose(0, 1)
        basic_read_info = basic_read_info.reshape(batch_size, -1)

        return basic_read_info, cm_read_info_1, cm_read_info_2

    # noise reduction cm read head (before div all data minus smallest noise)
    def cm_read_2(self, basic_read_matrix, cm_embedding,zero_add_vec):
        min_info, _ = basic_read_matrix.min(dim=-1, keepdim=True)
        basic_read_minus_min = basic_read_matrix - min_info
        basic_read_minus_min = torch.where(abs(basic_read_minus_min)<0.0001,torch.zeros_like(basic_read_minus_min) + 100000, basic_read_minus_min)
        cm_read = (basic_read_minus_min + zero_add_vec).div(cm_embedding)
        min_cm_read, _ = torch.min(cm_read, dim=-1)
        # min_cm_read_view = min_cm_read.view(min_cm_read.shape[1],-1)
        # min_cm_read_transpose = min_cm_read.transpose(0,1)
        min_info = min_info.squeeze().transpose(0, 1)
        min_cm_read = min_cm_read.transpose(0, 1)
        return torch.cat((min_info, min_cm_read), dim=-1)

    # cm_read_head
    def cm_read_1(self, basic_read_matrix, cm_embedding,zero_add_vec):
        cm_basic_read_matrix = basic_read_matrix + zero_add_vec
        cm_read = cm_basic_read_matrix.div(cm_embedding)
        # cm_read = cm_read.view(basic_read_matrix.shape[0], -1)
        min_cm_read, _ = cm_read.min(dim=-1)
        return min_cm_read.transpose(0, 1)

    def get_memory_usage(self):
        total_bytes = 0

        for param in self.parameters():
            total_bytes += param.nelement() * param.element_size()

        for buf in self.buffers():
            total_bytes += buf.nelement() * buf.element_size()

        return total_bytes
