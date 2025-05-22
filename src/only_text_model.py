"""
Adapted from: https://github.com/safe-graph/GNN-FakeNews
"""
from typing import Optional, overload
from typing_extensions import Self

import torch
import torch.nn.functional as F
# from adapters import BertAdapterModel
from multipledispatch import dispatch
from torch import Tensor, dtype
from torch._prims_common import DeviceLikeType
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader, Data
from transformers import BertTokenizer


class OnlyTextModel(torch.nn.Module):
    def __init__(self, args):
        super(OnlyTextModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_encoder_features = 768
        self.nhid = args.nhid
        self.num_classes = args.num_classes

        self.lin0 = torch.nn.Linear(self.num_encoder_features, self.nhid)

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.encoder = BertAdapterModel.from_pretrained('bert-base-uncased')
        #
        # # Load pre-trained task adapter from Adapter Hub
        # # This method call will also load a pre-trained classification head for the adapter task
        # adapter_name = self.encoder.load_adapter("sentiment/sst-2@ukp", config='pfeiffer')
        #
        # # Activate the adapter we just loaded, so that it is used in every forward pass
        # self.encoder.set_active_adapters(adapter_name)

        self.lin2 = torch.nn.Linear(self.nhid,1)#, self.num_classes)

    @dispatch(object, object, batch=Optional[Tensor])#, titles=Optional[list])
    def forward(self, x, edge_index, batch: Optional[Tensor] = None):#, titles: Optional[list] = None):
        input_x = x

        if batch is None:
            news = input_x[0].unsqueeze(0)
        else:
            nr_batches = batch.unique().shape[0]
            news = torch.stack([input_x[(batch == idx).nonzero().squeeze()[0]] for idx in range(nr_batches)])

        x = F.relu(self.lin0(news))
        x = self.lin2(x).view(-1)

        return x

    @dispatch(object)
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        return self.forward(x, edge_index, batch=batch) #, titles=data.title
