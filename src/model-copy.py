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


class Model(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_encoder_features = 768
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.encoder = args.model
        self.concat = concat

        if self.encoder == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.encoder == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.encoder == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)

        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_encoder_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

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

    @dispatch(object, object, data=Data)#, titles=Optional[list])
    def forward(self, x, edge_index, data: Optional[Data] = None):#, titles: Optional[list] = None):

        # if titles is None:
        #     titles = []

        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = gmp(x, data.batch)

        # if self.concat:
        #     cls_embeddings = self._encode(titles)
        #
        #     news = torch.stack(cls_embeddings)
        #     news = F.relu(self.lin0(news))
        #     x = torch.cat([x, news], dim=1)
        #     x = F.relu(self.lin1(x))

        if self.concat:
            # content = F.relu(self.lin0(content))
            # x = torch.cat([x, content], dim=1)
            # x = F.relu(self.lin1(x))

            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))


        # x = F.log_softmax(self.lin2(x), dim=-1)
        # x = F.sigmoid(self.lin2(x).view(-1))
        x = self.lin2(x).view(-1)
        # x = torch.tensor([1 - x.item(), x.item()])
        # x = torch.log(x)
        return x

    @dispatch(object)
    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        return self.forward(x, edge_index, data=data) #, titles=data.title
