'''
Generate necessary files
'''
import copy
import os
import json
import re
import csv
import gzip
import subprocess
import requests
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import scipy.sparse
from scipy import sparse
from scipy.sparse import coo_matrix
from requests.adapters import HTTPAdapter, Retry
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from tqdm import tqdm
from scipy.stats import normaltest


def extract_interaction_data(data_file):
    uniprot_id_list = set()  # store biogrid id or the protein in the network <str>
    interaction_list = set()  # store protein interactions in the network <tuple>

    with open(data_file) as f:
        next(f)
        biogrid_data = f.readlines()

    for line in tqdm(biogrid_data, desc='Extracting interaction information'):  # for line in biogrid_data:
        line = line.split('\t')
        # print(line)
        # print(len(line))
        if '0915' in line[11] or '0407' in line[11] or '0403' in line[11]:  # select interactionType
            id_1 = line[2].split('uniprot')
            id_2 = line[3].split('uniprot')
            if len(id_1) == 1 or len(id_2) == 1:
                continue
            uid_1, uid_2 = [], []
            for i in id_1:
                if '/swiss-prot:' in i:
                    uid_1.append(i.split(':')[1].split('|')[0])
            for i in id_2:
                if '/swiss-prot:' in i:
                    uid_2.append(i.split(':')[1].split('|')[0])
            for i1 in uid_1:
                for i2 in uid_2:
                    if i1 == i2:
                        continue
                    uniprot_id_list.add(i1)
                    uniprot_id_list.add(i2)
                    interaction_list.add((i1, i2))
                    interaction_list.add((i2, i1))
    # print(len(uniprot_id_list), 'len protein list')
    # print(len(interaction_list), 'len interaction list')

    uniprot_id_list = list(uniprot_id_list)  # dividing the same elements
    uniprot_id_list.sort()
    interaction_list = list(interaction_list)
    return_dict = {
        'id_list': uniprot_id_list,
        'interaction_list': interaction_list
    }

    return return_dict


def construct_uniprot_ppi(uniprot_list, interaction_list):
    node_set = set()
    idx_map = {}
    flag = 0
    for i in uniprot_list:
        idx_map[i] = flag
        flag += 1

    for interaction in tqdm(interaction_list, desc='record coordinates'):
        uni_1 = idx_map[interaction[0]]#uniprot_list.index(interaction[0])  # row
        uni_2 = idx_map[interaction[1]]#uniprot_list.index(interaction[1])  # col
        node_set.add((uni_1, uni_2))
        node_set.add((uni_2, uni_1))

    ppi_row = []
    ppi_col = []
    ppi_data = []
    for node in tqdm(node_set, desc='constructing PPI adjacency matrix'):
        x = node[0]
        y = node[1]
        ppi_row.append(x)
        ppi_col.append(y)
        ppi_data.append(1)

    ppi = coo_matrix((ppi_data, (ppi_row, ppi_col)), shape=(len(uniprot_list), len(uniprot_list)))
    ppi.setdiag(0)
    ppi.eliminate_zeros()

    return ppi


def construct_normal_ppi(data='../data/support_materials/BIOGRID-ORGANISM-Homo_sapiens-4.4.203.mitab.txt'):
    interaction_data = extract_interaction_data(data_file=data)
    ppi = construct_uniprot_ppi(uniprot_list=interaction_data['id_list'], interaction_list=interaction_data['interaction_list'])

    return ppi, interaction_data['id_list']


def construct_gcn_matrix(data, sample_list, protein_list):
    expr_set = pd.read_csv(data)
    extract_list = ['uniprot_id']
    extract_list.extend(sample_list)
    expr_data = pd.DataFrame(expr_set[extract_list]).sort_values('uniprot_id')
    # print(expr_data)

    # drop mask
    expr_protein = set(expr_data['uniprot_id'].values.tolist())
    ppi_protein = set(protein_list)
    and_res = expr_protein & ppi_protein
    drop_mask = expr_protein - and_res
    # print(len(expr_protein))
    # print(len(ppi_protein))
    # print(len(and_res))
    # print(len(drop_mask))
    # print('-'*10)

    # group by and aggrate same protein with 'mean'
    expr_data = expr_data.groupby(expr_data['uniprot_id']).agg('mean')
    # print(expr_data)
    # print('-' * 10)

    # remove data not in ppi
    for item in expr_data.index:
        if item in drop_mask:
            expr_data.drop(item, inplace=True)

    # add nodes
    nums = [[0] * 3 for i in range(len(protein_list))]
    expr_gcn = pd.DataFrame(data=nums, index=protein_list, columns=sample_list)
    for item in tqdm(expr_gcn.index, desc='protein expr'):
        if item in expr_data.index:
            expr_gcn.loc[item] = expr_data.loc[item]

    expr_gcn = expr_gcn.to_numpy()
    expr_pcc = np.corrcoef(expr_gcn)
    np.fill_diagonal(expr_pcc, 0)
    pcc_nan = np.isnan(expr_pcc)
    expr_pcc[pcc_nan] = 0
    gcn = coo_matrix(expr_pcc)

    return gcn, expr_gcn


def edge_clustering_coefficients(ppi_net, epsilon=0):
    ppi = ppi_net.tocsr()
    ecc_row = []
    ecc_col = []
    ecc_data = []
    for i in tqdm(range(ppi.shape[0]), desc='construct ecc matrix'):
        i_data = ppi[i].toarray()
        neighbors = ppi[i].indices
        degree_i = ppi[i].data.sum()
        for j in neighbors[neighbors > i]:
            j_data = ppi[j].toarray()
            triangles_num = np.logical_and(i_data, j_data).sum()
            degree_j = ppi[j].data.sum()
            possibly_triangles_num = min(degree_i, degree_j) - 1
            if possibly_triangles_num == 0:
                value = epsilon
            else:
                value = triangles_num / possibly_triangles_num

            ecc_row.append(i)
            ecc_col.append(j)
            ecc_data.append(value)
            ecc_row.append(j)
            ecc_col.append(i)
            ecc_data.append(value)

    ecc = coo_matrix((ecc_data, (ecc_row, ecc_col)), shape=ppi.shape)

    return ecc


def modify_network_topology(ppi_net, pcc_nor, pcc_inter):
    with tqdm(total=5, desc='modify protein interaction network') as mod_bar:
        # print(ppi_net.getnnz())
        ppi_net = ppi_net.tocsr()
        pcc_normal = pcc_nor.tocsr()
        pcc_interverion = pcc_inter.tocsr()
        mod_bar.update()
        diff_matrix = pcc_interverion - pcc_normal  # difference matrix

        diff_mat = diff_matrix.tocoo()
        scipy.sparse.save_npz('../diff', diff_mat)

        mod_bar.update()
        ppi_intervention = copy.deepcopy(ppi_net).todense()
        mod_bar.update()

        # compute thresholds
        diff_matrix = diff_matrix.toarray()
        diff_std = np.std(diff_matrix)
        diff_mean = np.mean(diff_matrix)
        l_threshold = diff_mean - 2.75 * diff_std
        r_threshold = diff_mean + 2.75 * diff_std
        mod_bar.update()
        # modify topology
        res1 = np.logical_and(diff_matrix < l_threshold, ppi_intervention == 1).A
        # print(res1.any())
        # res_rmv = coo_matrix(res1.astype(int))
        # scipy.sparse.save_npz('../rmv', res_rmv)
        # print(res1.astype(int).sum(), '< and 1')
        # print((diff_matrix < l_threshold).astype(int).sum(), '<')
        # print((ppi_intervention == 1).astype(int).sum(), '1')
        res2 = np.logical_and(diff_matrix > r_threshold, ppi_intervention == 0).A
        # print(res2.any())
        # res_add = coo_matrix(res2.astype(int))
        # scipy.sparse.save_npz('../add', res_add)
        # print(res2.astype(int).sum(), '> and 0')
        # print((diff_matrix > r_threshold).astype(int).sum(), '>')
        # print((ppi_intervention == 0).astype(int).sum(), '0')

        ppi_intervention[res1] = 0
        ppi_intervention[res2] = 1
        # res1 = diff_matrix < l_threshold
        # res2 = ppi_intervention == 1
        # print(res1.any(), res2.any())

        ppi_intervention = coo_matrix(ppi_intervention)
        mod_bar.update()
    mod_bar.close()

    return ppi_intervention

def construct_matrix_of_normal_and_intervention_cond(data):
    normal_path = '../data/generate_materials/'
    protein_list_path = normal_path + 'protein_ppi.json'

    if not os.path.exists(normal_path + 'PPI_normal.npz'):
        ppi_normal, protein_list = construct_normal_ppi()
        with tqdm(total=2, desc='PPI normal & protein files store') as s_bar:
            sparse.save_npz(normal_path + 'PPI_normal', ppi_normal)
            s_bar.update()
            if not Path(protein_list_path).exists():
                with open(protein_list_path, 'w') as f:
                    json.dump(protein_list, f)
                s_bar.update()
    else:
        with tqdm(total=2, desc='PPI normal & protein files load') as s_bar:
            ppi_normal = sparse.load_npz(normal_path + 'PPI_normal.npz')
            s_bar.update()
            with open(protein_list_path) as f:
                protein_list = json.load(f)
            s_bar.update()

    if not os.path.exists(normal_path + 'ECC_normal.npz'):
        ecc_normal = edge_clustering_coefficients(ppi_net=ppi_normal)
        with tqdm(total=1, desc='ECC normal file store') as s_bar:
            sparse.save_npz(normal_path + 'ECC_normal', ecc_normal)
            s_bar.update()

    for sample_data in data.values():
        if not os.path.exists(normal_path + 'GCN_normal.npz'):
            gcn_normal, expr_normal = construct_gcn_matrix(sample_data[1], sample_data[2]['normal'], protein_list)
            with tqdm(total=2, desc='GCN normal & expr files store') as s_bar:
                sparse.save_npz(normal_path + 'GCN_normal', gcn_normal)
                s_bar.update()
                if not os.path.exists(normal_path + 'expr_normal'):
                    np.save(normal_path + 'expr_normal', expr_normal)
                    s_bar.update()
        else:
            with tqdm(total=1, desc='GCN normal file load') as s_bar:
                gcn_normal = sparse.load_npz(normal_path + 'GCN_normal.npz')
                s_bar.update()

        series = sample_data[1].split('/')[-1].split('_')[0]
        inter_path = '../data/generate_materials/' + series + '_data/'
        if not os.path.exists(inter_path):
            os.makedirs(inter_path)

        if not os.path.exists(inter_path + 'GCN_inter.npz'):
            gcn_inter, expr_inter = construct_gcn_matrix(sample_data[1], sample_data[2]['intervention'], protein_list)
            with tqdm(total=2, desc='GCN inter & expr files store') as s_bar:
                sparse.save_npz(inter_path + 'GCN_inter', gcn_inter)
                s_bar.update()
                if not os.path.exists(inter_path + 'expr_inter'):
                    np.save(inter_path + 'expr_inter', expr_inter)
                s_bar.update()
        else:
            with tqdm(total=1, desc='GCN inter file load') as s_bar:
                gcn_inter = sparse.load_npz(inter_path + 'GCN_inter.npz')
                s_bar.update()

        ppi_inter = modify_network_topology(ppi_net=ppi_normal, pcc_nor=gcn_normal, pcc_inter=gcn_inter)
        if not os.path.exists(inter_path + 'PPI_inter.npz'):
            sparse.save_npz(inter_path + 'PPI_inter', ppi_inter)
            print('PPI_inter saved')

        ecc_inter = edge_clustering_coefficients(ppi_net=ppi_inter)
        if not os.path.exists(inter_path + 'ECC_inter.npz'):
            sparse.save_npz(inter_path + 'ECC_inter', ecc_inter)
            print('ECC_inter saved')


def judge_gene_onthology_line(line, go_list):
    if line.startswith('DR   GO;') and 'C:' in line and ('IDA' in line or 'HDA' in line or 'IEA' in line or 'EXP' in line or 'IPI' in line) and line[9:19] in go_list:
        return True
    else:
        return False


def extract_localization_data(uniprot_sprot_data='../data/support_materials/uniprot_sprot_human.dat.gz'):
    with tqdm(total=3, desc='cellular component data reading') as loc_bar:
        with gzip.open(uniprot_sprot_data) as f:
            data = f.read().decode()
        entry_list = data.split('//\n')[0: -1]  # split each protein data
        loc_dict = {}  # protein and corresponding CC
        loc_bar.update()

        with open('../data/support_materials/cellular_component.txt') as f:
            loc_list = f.read().split()
        loc_bar.update()

        with open('../data/generate_materials/protein_ppi.json') as f:
            uni_list = json.load(f)
        loc_bar.update()
    loc_bar.close()

    for entry in tqdm(entry_list, desc='cellular component data extracting'):
        AC, CC_list = None, []
        lines = entry.split('\n')
        for line in lines:
            if not AC:
                if line.startswith('AC'):
                    AC = line.split()[1].replace(';', '')
            elif judge_gene_onthology_line(line, loc_list):
                CC = line[9: 19]
                CC_list.append(CC)
        if AC in uni_list and CC_list:
            loc_dict[AC] = CC_list

    label_list = []
    for item in uni_list:
        if item in loc_dict.keys():
            loc = loc_dict[item]
        else:
            loc = []
        label_item = (item, loc)
        label_list.append(label_item)

    return label_list


def construct_protein_loc_matrix(label_list):
    """
    Construction of protein localization annotation matrix

    :param vir_loc_list: Virtual positioning protein list.
    :param file: Generate files
    :return: protein localization annotation matrix
    """
    with open('../data/support_materials/cellular_component.txt') as f:
        loc_list = f.read().split()

    ncol = len(loc_list)
    nrow = len(label_list)

    loc_row = []
    loc_col = []
    loc_data = []
    protein_list, loc = zip(*label_list)

    for protein, localizations in tqdm(label_list, desc='construct loc matrix'):
        row = protein_list.index(protein)
        if localizations:
            for localization in localizations:
                col = loc_list.index(localization)
                data = 1.
                loc_row.append(row)
                loc_col.append(col)
                loc_data.append(data)

    loc_matrix = coo_matrix((loc_data, (loc_row, loc_col)), shape=(nrow, ncol))

    return loc_matrix


def construct_loc_matrix():
    label_list = extract_localization_data()
    loc_matrix = construct_protein_loc_matrix(label_list)
    label_with_loc = extract_data_with_position(label_list)

    sparse.save_npz('../data/generate_materials/loc_matrix', loc_matrix)
    label_with_loc_path = '../data/generate_materials/label_with_loc_list.json'
    with open(label_with_loc_path, 'w') as f:
        json.dump(label_with_loc, f)
    label_path = '../data/generate_materials/label_list.json'
    with open(label_path, 'w') as f:
        json.dump(label_list, f)




# def construct_graph_data(ppi, gcn, label, vir_label):
#     interaction_idx = list(zip(ppi.row, ppi.col))
#     protein = list(list(zip(*label))[0])
#     vir_labal = list(list(zip(*vir_label))[0])
#     interaction = set()
#     gcn = gcn.toarray()
#
#     for item in tqdm(interaction_idx, desc='add virtual protein'):
#         uni_a = protein[item[0]]
#         uni_b = protein[item[1]]
#         loc_a = label[item[0]][1]
#         loc_b = label[item[1]][1]
#         expr_row = protein.index(uni_a)
#         expr_col = protein.index(uni_b)
#         expr_data = gcn[expr_row][expr_col]
#
#         loc_a_flag = True if len(loc_a) > 1 else False
#         loc_b_flag = True if len(loc_b) > 1 else False
#
#         if loc_a_flag and loc_b_flag:
#             for i in range(len(loc_a)):
#                 for j in range(len(loc_b)):
#                     a = uni_a + '_' + str(i)
#                     b = uni_b + '_' + str(j)
#                     a_idx = vir_labal.index(a)
#                     b_idx = vir_labal.index(b)
#                     inter = (a_idx, b_idx, expr_data)
#                     interaction.add(inter)
#
#         elif loc_a_flag or loc_b_flag:
#             if loc_a_flag:
#                 for i in range(len(loc_a)):
#                     a = uni_a + '_' + str(i)
#                     b = uni_b
#                     a_idx = vir_labal.index(a)
#                     b_idx = vir_labal.index(b)
#                     inter = (a_idx, b_idx, expr_data)
#                     interaction.add(inter)
#
#             elif loc_b_flag:
#                 for j in range(len(loc_b)):
#                     a = uni_a
#                     b = uni_b + '_' + str(j)
#                     a_idx = vir_labal.index(a)
#                     b_idx = vir_labal.index(b)
#                     inter = (a_idx, b_idx, expr_data)
#                     interaction.add(inter)
#
#         else:
#             a = uni_a
#             b = uni_b
#             a_idx = vir_labal.index(a)
#             b_idx = vir_labal.index(b)
#             inter = (a_idx, b_idx, expr_data)
#             interaction.add(inter)
#
#     ppi_row = []
#     ppi_col = []
#     ppi_data = []
#     gcn_row = []
#     gcn_col = []
#     gcn_data = []
#     for item in tqdm(interaction, desc='construct ppi matrix and gcn matrix'):
#         ppi_row.append(item[0])
#         ppi_col.append(item[1])
#         ppi_data.append(1)
#         gcn_row.append(item[0])
#         gcn_col.append(item[1])
#         gcn_data.append(item[2])
#
#     ppi_mat = coo_matrix((ppi_data, (ppi_col, ppi_row)), shape=(len(vir_labal), len(vir_labal)))
#     gcn_mat = coo_matrix((gcn_data, (gcn_row, gcn_col)), shape=(len(vir_labal), len(vir_labal)))
#     ppi_mat.setdiag(0)
#     gcn_mat.setdiag(0)
#     ppi_mat.eliminate_zeros()
#     gcn_mat.eliminate_zeros()
#
#     ecc_mat = edge_clustering_coefficients(ppi_mat)
#
#     return ppi_mat, ecc_mat, gcn_mat


def extract_data_with_position(label_list):
    uni_list, loc_list = zip(*label_list)
    uni_idx = []
    for item in label_list:
        if item[1]:
            uni = item[0]
            idx = uni_list.index(uni)
            uni_idx.append(idx)

    return uni_idx


if __name__ == '__main__':
    data_dict = {
        1: {
            1: '../data/support_materials/GSE30931_exprSet.csv',
            2: {
                'normal': ['GSM766676', 'GSM766677', 'GSM766678'],
                'intervention': ['GSM766682', 'GSM766683', 'GSM766684']
            }
        },
    }

    construct_matrix_of_normal_and_intervention_cond(data_dict)
    construct_loc_matrix()

    # # check
    # ppi = sparse.load_npz('../data/generate_materials/PPI_normal.npz')
    # ecc = sparse.load_npz('../data/generate_materials/ECC_normal.npz')
    # gcn = sparse.load_npz('../data/generate_materials/GCN_normal.npz')
    # print(ppi.shape, ecc.shape, gcn.shape)











