'''
Utils
'''
import dgl
import random
import torch
import torch as th
import json
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from scipy.sparse import load_npz, coo_matrix, save_npz


def data_normalize(mat):
    mat_normalize = mat.copy()
    p_mean = mat_normalize.mean(0)
    p_std = mat_normalize.std(0)
    for j in range(mat_normalize.shape[1]):
        mat_normalize[:, j] = (mat_normalize[:, j] - p_mean[j]) / p_std[j]

    return mat_normalize


def create_graph(ppi, ecc, gcn, loc, expr, uniprot):
    start = list(ppi.row)
    end = list(ppi.col)
    num_nodes = len(uniprot)
    g = dgl.graph((start, end), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    g.nodes[list(range(num_nodes))].data['loc'] = th.from_numpy(loc.toarray().astype(np.float32))
    edge_feat = np.hstack((ecc, gcn))
    node_feat = th.tensor(np.hstack((expr, edge_feat)), dtype=th.float)
    g.nodes[list(range(num_nodes))].data['feat'] = node_feat

    return g


def pca(mat, components):
    pca = PCA(n_components=components, random_state=42)
    new_node_feat = pca.fit_transform(mat)

    return new_node_feat


def smote(loc, ppi, scale=0.7):
    # determine class thresholds
    loc_mat = loc.toarray()
    loc_sum = loc_mat.sum(axis=0).astype(int)
    print('loc sum:\t', loc_sum)
    large_threshold = loc_sum.max() - 0.4 * (loc_sum.max() - loc_sum.min())
    minor_threshold = loc_sum.min() + 0.1 * (loc_sum.max() - loc_sum.min())
    print('large threshold:\t', large_threshold)
    print('minor threshold:\t', minor_threshold)
    large_mask = [1 if i > large_threshold else 0 for i in loc_sum]
    minor_mask = [1 if i < minor_threshold else 0 for i in loc_sum]
    print('large mask:\t', large_mask)
    print('minor mask:\t', minor_mask)
    # statistical major and minor classes (store index)
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
    intermediate_mask = np.logical_not(np.logical_or(large_mask, minor_mask)).astype(int).tolist()  # nor_num class mask
    intermediate = [loc_sum[i] if intermediate_mask[i] else None for i in range(len(loc_sum))]  # nor_num class number
    intermediate = list(filter(None, intermediate))
    inter_mean = int(np.mean(intermediate))
    print('inter mean:\t', inter_mean)
    large_num = [int(scale * (loc_sum[i] - inter_mean)) if large_mask[i] else 0 for i in range(len(loc_sum))]
    minor_num = [int(scale * (inter_mean - loc_sum[i])) if minor_mask[i] else 0 for i in range(len(loc_sum))]
    print('large num:\t', large_num)
    print('minor num:\t', minor_num)
    # remove large class nodes
    loc_new = loc.toarray()
    for col_idx in range(len(large_num)):
        if large_num[col_idx]:
            col = loc_new[:, col_idx:col_idx+1]
            row_idx = np.where(col == 1)[0].tolist()
            remove_idx = random.sample(row_idx, large_num[col_idx])
            loc_new[remove_idx, col_idx] = 0
    new_loc_sum = loc_new.sum(axis=0).astype(int)

    with_label = np.where(loc_new.any(axis=1))[0].tolist()
    with open('../data/generate_materials/label_with_loc_list_new.json', 'w') as f:
        json.dump(with_label, f)

    print('new loc sum:\t', new_loc_sum)
    loc_new = coo_matrix(loc_new)
    save_npz('../data/generate_materials/loc_matrix_new.npz', loc_new)

if __name__ == '__main__':
    # delete node
    # ppi = load_npz('../data/generate_materials/PPI_normal.npz')
    # loc = load_npz('../data/generate_materials/loc_matrix.npz')
    # smote(loc, ppi)


    # PCA
    ppi = load_npz('../data/generate_materials/PPI_normal.npz')
    gcn = load_npz('../data/generate_materials/GCN_normal.npz').tocsr().multiply(ppi.tocsr()).toarray()
    ecc = load_npz('../data/generate_materials/ECC_normal.npz').toarray()
    ecc_pca = pca(ecc, 250)
    gcn_pca = pca(gcn, 250)
    np.save('../data/generate_materials/ECC_normal_pca', ecc_pca)
    np.save('../data/generate_materials/GCN_normal_pca', gcn_pca)

    ppi_inter = load_npz('../data/generate_materials/GSE30931_data/PPI_inter.npz')
    gcn_inter = load_npz('../data/generate_materials/GSE30931_data/GCN_inter.npz').tocsr().multiply(ppi_inter.tocsr()).toarray()
    ecc_inter = load_npz('../data/generate_materials/GSE30931_data/ECC_inter.npz').toarray()
    ecc_inter_pca = pca(ecc_inter, 250)
    gcn_inter_pca = pca(gcn_inter, 250)
    np.save('../data/generate_materials/GSE30931_data/ECC_inter_pca', ecc_inter_pca)
    np.save('../data/generate_materials/GSE30931_data/GCN_inter_pca', gcn_inter_pca)



