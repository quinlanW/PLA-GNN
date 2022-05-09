import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class GNN11(nn.Module):
    def __init__(self, in_feats, h1_feats, num_classes, dropout=0.5):
        super(GNN11, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, 'pool')
        self.liner1 = nn.Linear(h1_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.liner1(h)
        h = th.sigmoid(h)

        return h


class GNN12(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes, dropout=0.5):
        super(GNN12, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, 'pool')
        self.liner1 = nn.Linear(h1_feats, h2_feats)
        self.drop1 = nn.Dropout(p=dropout)
        self.liner2 = nn.Linear(h2_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        # h = F.relu(h)
        h = self.liner1(h)
        h = F.leaky_relu(h)
        # h = F.relu(h)
        h = self.drop1(h)
        h = self.liner2(h)
        h = th.sigmoid(h)

        return h


class GNN21(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes, dropout=0.5):
        super(GNN21, self).__init__()
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


class GNN22(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, h3_feats, num_classes, dropout=0.5):
        super(GNN22, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, 'pool')
        self.conv2 = SAGEConv(h1_feats, h2_feats, 'pool')
        self.liner1 = nn.Linear(h2_feats, h3_feats)
        self.liner2 = nn.Linear(h3_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.liner1(h)
        h = F.leaky_relu(h)
        h = self.liner2(h)
        h = th.sigmoid(h)

        return h


class GNN31(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, h3_feats, num_classes, dropout=0.5):
        super(GNN31, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, 'pool')
        self.conv2 = SAGEConv(h1_feats, h2_feats, 'pool')
        self.conv3 = SAGEConv(h2_feats, h3_feats, 'pool')
        self.liner1 = nn.Linear(h3_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.conv3(g, h)
        h = F.leaky_relu(h)
        h = self.liner1(h)
        h = th.sigmoid(h)

        return h


class GNN32(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, h3_feats, h4_feats, num_classes, dropout=0.5):
        super(GNN32, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, 'pool')
        self.conv2 = SAGEConv(h1_feats, h2_feats, 'pool')
        self.conv3 = SAGEConv(h2_feats, h3_feats, 'pool')
        self.liner1 = nn.Linear(h3_feats, h4_feats)
        self.liner2 = nn.Linear(h4_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.conv3(g, h)
        h = F.leaky_relu(h)
        h = self.liner1(h)
        h = F.leaky_relu(h)
        h = self.liner2(h)
        h = th.sigmoid(h)

        return h