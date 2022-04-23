import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class GNN1(nn.Module):
    def __init__(self, in_feats, h1_feats, num_classes):
        super(GNN1, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, 'pool')
        self.liner1 = nn.Linear(h1_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.liner1(h)
        h = th.sigmoid(h)

        return h


class GNN2(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes):
        super(GNN2, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, 'pool')
        self.conv2 = SAGEConv(h1_feats, h2_feats, 'pool')
        self.liner1 = nn.Linear(h2_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.liner1(h)
        h = th.sigmoid(h)

        return h
