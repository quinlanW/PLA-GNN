import dgl
import random
import torch
import torch as th
import json
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from scipy.sparse import load_npz


def create_graph(ppi, ecc, gcn, loc, uniprot):
    start = list(ppi.row)
    end = list(ppi.col)
    num_nodes = len(uniprot)
    g = dgl.graph((start, end), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    g.nodes[list(range(num_nodes))].data['loc'] = th.from_numpy(loc.toarray().astype(np.float32))
    node_feat = th.tensor(np.hstack((ecc, gcn)), dtype=th.float)
    g.nodes[list(range(num_nodes))].data['feat'] = node_feat

    return g


def pca(mat):
    pca = PCA(n_components=2500, random_state=42)
    new_node_feat = pca.fit_transform(mat)

    return new_node_feat


def smote(loc, ppi, scale=0.5):
    # determine class thresholds
    loc_mat = loc.toarray()
    loc_sum = loc_mat.sum(axis=0).astype(int)
    large_threshold = loc_sum.max() - 0.1 * (loc_sum.max() - loc_sum.min())
    minor_threshold = loc_sum.min() + 0.1 * (loc_sum.max() - loc_sum.min())
    large_mask = [1 if i > large_threshold else 0 for i in loc_sum]
    minor_mask = [1 if i < minor_threshold else 0 for i in loc_sum]
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
    large_num = [int(scale * (loc_sum[i] - inter_mean)) if large_mask[i] else 0 for i in range(len(loc_sum))]
    minor_num = [int(scale * (inter_mean - loc_sum[i])) if minor_mask[i] else 0 for i in range(len(loc_sum))]
    # remove large class nodes
    # add minor class nodes
    ppi_mat = ppi.toarray()
    adj = nx.from_numpy_matrix(ppi_mat)
    break_flag = np.array(minor_mask)
    while True:
        # break opt
        if np.all(break_flag >= minor_num) or np.any(break_flag >= inter_mean) or break_flag.sum()/np.sum(minor_mask) > np.sum(minor_num)/np.sum(minor_mask):
            break
        # choose source node and target node
        source = minor[random.randint(0, len(minor)-1)]
        random.shuffle(minor)
        min_dist = float('inf')
        target = source
        for target_idx in minor:
            dist = len(nx.shortest_path(adj, source, target_idx))
            if dist != 1 and dist < min_dist:
                min_dist = dist
                target = target_idx
        # mix node label
        mix_label = np.logical_or(loc_mat[source], loc_mat[target]).astype(int)
        req_label = np.logical_and(mix_label, minor_mask).astype(int)
        ava_label = np.logical_xor(mix_label, req_label).astype(int)
        # mix node edge
        s_neighbors = list(adj.neighbors(source))
        t_neighbors = list(adj.neighbors(target))

        # print(min_dist, target, source)
        # print(len(s_neighbors))
        # print(len(t_neighbors))
        # print(loc_mat[source].astype(int), " source")
        # print(loc_mat[target].astype(int), " target")
        # print(mix_label,  " mix")
        # print(req_label, " req")
        # print(ava_label, " ava")

        break_flag += req_label
        print(break_flag)

        # break

    # print('loc sum:\t', loc_sum)
    # print('large mask:\t', large_mask)
    # print('minor mask:\t', minor_mask)
    # print('inter mean:\t', inter_mean)
    # print('large num:\t', large_num)
    # print('minor num:\t', minor_num)
    # print(len(minor), len(both), len(large))


if __name__ == '__main__':
    ppi = load_npz('../data/generate_materials/PPI_normal.npz')
    gcn = load_npz('../data/generate_materials/GCN_normal.npz').tocsr().multiply(ppi.tocsr()).toarray()
    ecc = load_npz('../data/generate_materials/ECC_normal.npz').toarray()
    ecc_pca = pca(ecc)
    gcn_pca = pca(gcn)
    np.save('../data/generate_materials/ECC_normal_pca', ecc_pca)
    np.save('../data/generate_materials/GCN_normal_pca', gcn_pca)

    ppi_inter = load_npz('../data/generate_materials/GSE30931_data/PPI_inter.npz')
    gcn_inter = load_npz('../data/generate_materials/GSE30931_data/GCN_inter.npz').tocsr().multiply(ppi.tocsr()).toarray()
    ecc_inter = load_npz('../data/generate_materials/GSE30931_data/ECC_inter.npz').toarray()
    ecc_inter_pca = pca(ecc_inter)
    gcn_inter_pca = pca(gcn_inter)
    np.save('../data/generate_materials/ECC_inter_pca', ecc_inter_pca)
    np.save('../data/generate_materials/GCN_inter_pca', gcn_inter_pca)



