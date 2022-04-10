import dgl
import torch as th
import json
import numpy as np
from scipy.sparse import load_npz


def create_graph(ppi, ecc, gcn, loc, uniprot):
    start = list(ppi.row)
    end = list(ppi.col)
    num_nodes = len(uniprot)
    g = dgl.graph((start, end), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    g.edges[list(ecc.row), list(ecc.col)].data['ecc'] = th.Tensor(list(ecc.data))
    g.edges[list(gcn.row), list(gcn.col)].data['gcn'] = th.Tensor(list(gcn.data))
    g.nodes[list(range(num_nodes))].data['loc'] = th.from_numpy(loc.toarray().astype(np.float))
    g.nodes[list(range(num_nodes))].data['feat'] = th.eye(num_nodes)
    # g.nodes[list(range(num_nodes))].data['feat'] = th.from_numpy(gcn.toarray()).long()

    return g


# if __name__ == "__main__":
#     PPI = load_npz('../data/generate_materials/PPI_normal.npz')
#     GCN = load_npz('../data/generate_materials/GCN_normal.npz')
#     ECC = load_npz('../data/generate_materials/ECC_normal.npz')
#     LOC = load_npz('../data/generate_materials/loc_matrix.npz')
#     with open('../data/generate_materials/protein_ppi.json', 'r') as f:
#         uniprot = json.load(f)
#     g = create_graph(PPI, ECC, GCN, LOC, uniprot)
#     print(g)

    # add virtual protein
    # PPI = load_npz('../data/generate_materials/vir_PPI_normal.npz')
    # GCN = load_npz('../data/generate_materials/vir_GCN_normal.npz')
    # ECC = load_npz('../data/generate_materials/vir_ECC_normal.npz')
    # LOC = load_npz('../data/generate_materials/vir_loc_matrix.npz')
    # with open('../data/generate_materials/vir_label_list.json', 'r') as f:
    #     uniprot = json.load(f)
    # uniprot = list(list(zip(*uniprot))[0])
    # g = create_graph(PPI, ECC, GCN, LOC, uniprot)
    # print(g)



    # cord = list(zip(ppi.row, ppi.col))
    # print(cord)  # jianchan shifoushi shuangxiangbian