import numpy as np

from model import *
from train import *
from utils import *
from torch.nn import MultiLabelSoftMarginLoss


ppi = load_npz('../data/generate_materials/PPI_normal.npz')
gcn = load_npz('../data/generate_materials/GCN_normal.npz').tocsr().multiply(ppi.tocsr()).tocoo()
ecc = load_npz('../data/generate_materials/ECC_normal.npz')
loc = load_npz('../data/generate_materials/loc_matrix.npz')
with open('../data/generate_materials/protein_ppi.json', 'r') as f:
    uniprot = json.load(f)

loc_mat = load_npz('../data/generate_materials/loc_matrix.npz').toarray()
loc_num = loc_mat.sum(axis=0)
weight = th.Tensor(loc_num.sum() / loc_num)

# torch.cuda.set_device(1)
# print(torch.cuda.current_device())
# device = 'cuda:1'
device = 'cpu'

g = create_graph(ppi, ecc, gcn, loc, uniprot)
g = g.to(device)
criterion = MultiLabelSoftMarginLoss(weight=weight)
for alp in [0.2, 0.3, 0.1]:  # , 0.3]:
    train(g, criterion, lr=0.00005, alpha=alp, device=device)


