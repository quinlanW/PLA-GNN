import dgl
import torch
import torch as th
# th.set_default_tensor_type(torch.DoubleTensor)
import json
import numpy as np
from scipy.sparse import load_npz


def create_graph(ppi, ecc, gcn, loc, uniprot):
    start = list(ppi.row)
    end = list(ppi.col)
    num_nodes = len(uniprot)
    g = dgl.graph((start, end), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    g.nodes[list(range(num_nodes))].data['loc'] = th.from_numpy(loc.toarray().astype(np.float32))

    # 1：1
    node_feat = ecc.tocsr().multiply(gcn.tocsr()).toarray()
    node_feat = th.tensor(node_feat, dtype=th.float)
    g.nodes[list(range(num_nodes))].data['feat'] = node_feat

    return g


def smote(loc, scale=0.5):
    # determine class thresholds
    loc_mat = loc.toarray()
    loc_sum = loc_mat.sum(axis=0).astype(int)
    large_threshold = loc_sum.max() - 0.1 * (loc_sum.max() - loc_sum.min())
    minor_threshold = loc_sum.min() + 0.1 * (loc_sum.max() - loc_sum.min())
    large_mask = [1 if i > large_threshold else 0 for i in loc_sum]
    minor_mask = [1 if i < minor_threshold else 0 for i in loc_sum]
    # statistical major and minor classes
    minor = []
    large = []
    both = []
    for locs_idx in range(len(loc_mat)):
        locs = loc_mat[locs_idx]
        if locs.sum() == 0:
            continue
        minor_res = np.logical_and(locs, minor_mask).astype(int)
        large_res = np.logical_and(locs, large_mask).astype(int)
        if minor_res.sum() != 0 and large_res.sum() != 0:
            both.append(locs_idx)
        else:
            if minor_res.sum() != 0:
                minor.append(locs_idx)
            if large_res.sum() != 0:
                large.append(locs_idx)
    # determine generate and remove nodes num
    intermediate_mask = np.logical_not(np.logical_or(large_mask, minor_mask)).astype(int).tolist()
    intermediate = [loc_sum[i] if intermediate_mask[i] else None for i in range(len(loc_sum))]
    intermediate = list(filter(None, intermediate))
    inter_mean = int(np.mean(intermediate))
    large_num = [int(scale * (loc_sum[i] - inter_mean)) if large_mask[i] else 0 for i in range(len(loc_sum))]
    minor_num = [int(scale * (inter_mean - loc_sum[i])) if minor_mask[i] else 0 for i in range(len(loc_sum))]
    # remove large class nodes



# if __name__ == '__main__':
#     loc_mat = load_npz('../data/generate_materials/loc_matrix.npz')
#     smote(loc_mat)


