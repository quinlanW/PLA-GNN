import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_feats, h1_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, 'pool')  # GraphConv(in_feats, h1_feats, allow_zero_in_degree=True)
        self.conv2 = SAGEConv(h1_feats, num_classes, 'pool')  # GraphConv(h1_feats, h2_feats, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)

        return h

