import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h1_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h1_feats, h2_feats, allow_zero_in_degree=True)
        # self.liner1 = nn.Linear(h2_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        # h = F.leaky_relu(h)
        # h = F.relu(h)
        h = th.sigmoid(h)
        h = self.conv2(g, h)
        # h = F.leaky_relu(h)
        # h = F.relu(h)
        # h = self.liner1(h)
        # h = th.sigmoid(h)  # 最终类别区分不明显，导致很多1
        # h = F.leaky_relu(h)  # 效果奇差无比
        # h = F.softmax(h, dim=1)  # 效果奇差无比

        return h

