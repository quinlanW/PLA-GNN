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


if __name__ == '__main__':
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



