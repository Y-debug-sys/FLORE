import torch

from torch import nn
from Baselines.meta_learning_sketch.loss_functions import LossFunc_for_MSE_ARE, LossFunc_for_ARE_AAE


class NeuralStructure(nn.Module):

    def __init__(
        self, 
        attention_matrix, 
        embedding_net, 
        decode_net,
        refine_net, 
        memory_matrix
    ):
        super(NeuralStructure, self).__init__()
        self.refine_net = refine_net
        self.embedding_net = embedding_net
        self.memory_matrix = memory_matrix
        self.decode_net = decode_net
        self.attention_matrix = attention_matrix

    def write(self, input_x, input_y):
        embedding = self.get_embedding(input_x)
        refined = self.get_refined(embedding)
        address = self.get_address(refined)
        self.memory_matrix.write(address, embedding, input_y)

    def clear(self):
        self.memory_matrix.clear()

    def normalize_attention_matrix(self):
        self.attention_matrix.normalize()

    # !!! attention !!!
    def get_embedding(self, input_x):
        embedding = self.embedding_net(input_x)
        # embedding = embedding / embedding.sum(dim=-1,keepdim=True)
        return embedding

    def get_refined(self, embedding):
        refined = self.refine_net(embedding)
        return refined

    def get_address(self, refined):
        address = self.attention_matrix(refined)
        return address

    def query(self, input_x, stream_length=None):
        embedding = self.get_embedding(input_x)
        refined = self.get_refined(embedding)
        address = self.get_address(refined)
        # read_info = self.memory_matrix.read(address, embedding)
        # decode_info = torch.cat((read_info, embedding, stream_length), dim=1)
        decode_info = self.memory_matrix.read(address, embedding)
        weight_pred = self.decode_net(decode_info)
        return weight_pred
    
    def get_memory_usage(self):
        return self.memory_matrix.get_memory_usage()


class MetaSketch:

    def __init__(
        self,
        model,
        optimizer=None, 
        learning_rate=1e-3,
        logger=None
    ):
        super(MetaSketch, self).__init__()
        self.logger, self.model = logger, model
        self.optimizer = optimizer if optimizer is not None else \
                         torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        self.loss_func = LossFunc_for_MSE_ARE()  # Optional: LossFunc_for_ARE_AAE, LossFunc_for_MSE_ARE

    def get_memory_usage(self):
        return self.model.get_memory_usage()

    def loss(self, pred, label):
        return self.loss_func(pred, label)
    
    def save(self, path):
        pass

    def train(self, train_loader, train_step): 
        for i in range(train_step): 
            self.reset()
            support_x, support_y, query_x, query_y = next(train_loader)
            self.model.write(support_x, support_y)

            # with torch.no_grad():
            #     stream_length = support_y.sum()

            query_pred = self.model.query(query_x)
            
            self.optimizer.zero_grad()
            loss = self.loss_func(query_pred, query_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            torch.cuda.empty_cache()
            self.model.normalize_attention_matrix()

    def insert(self, keys: torch.Tensor, values: torch.Tensor):
        self.model.write(keys, values)

    def query(self, keys: torch.Tensor):
        return self.model.query(keys)

    def reset(self):
        self.model.clear()
