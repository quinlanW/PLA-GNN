'''
Mislocalization log

For generating predictions of protein mislocalization under drug intervention
'''
import csv
import os
import glob
import numpy as np
import json
from tqdm import tqdm
from scipy.sparse import load_npz


def scaling(logit_mat):
    mat = logit_mat.copy()
    p_min = mat.min(0)
    for j in range(mat.shape[1]):
        mat[:, j] = mat[:, j] - p_min[j]

    p_max = mat.max(0)
    for j in range(mat.shape[1]):
        mat[:, j] = mat[:, j] / p_max[j]

    sum_row = mat.sum(1)
    for i in range(mat.shape[0]):
        mat[i] /= sum_row[i]

    return mat


def mat_merge(file_path="../data/log"):
    states = ['normal', 'perturbation']
    for paths in glob.glob(file_path+'/GSE*'):
        for state in states:
            path = paths + '/' + state
            res_path = path.split(state)[0].replace('log', 'res')
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            mat_cnt = np.zeros((24041, 12))

            for path in sorted(glob.glob(path + '/*.npy')):
                mat = np.load(path)
                mat = scaling(mat)
                mat_cnt += mat
                # print(path)
            mat_cnt /= 100
            np.save(res_path + state + '_logits.npy', mat_cnt)


def misloc_protein_record(normal_mat, inter_mat, data, threshold=100):
    # loc mapping
    loc_map = {
        'GO:0005938': 'Cell cortex',
        'GO:0005829': 'Cytosol',
        'GO:0015629': 'Actin cytoskeleton',
        'GO:0005794': 'Golgi apparatus',
        'GO:0005783': 'Endoplasmic reticulum',
        'GO:0005730': 'Nucleolus',
        'GO:0005777': 'Peroxisome',
        'GO:0005739': 'Mitochondrion',
        'GO:0005764': 'Lysosome',
        'GO:0005813': 'Centrosome',
        'GO:0005634': 'Nucleus',
        'GO:0005886': 'Plasma membrane',
        0: 'Cell cortex',
        1: 'Cytosol',
        2: 'Actin cytoskeleton',
        3: 'Golgi apparatus',
        4: 'Endoplasmic reticulum',
        5: 'Nucleolus',
        6: 'Peroxisome',
        7: 'Mitochondrion',
        8: 'Lysosome',
        9: 'Centrosome',
        10: 'Nucleus',
        11: 'Plasma membrane'
    }

    normal = scaling(normal_mat)
    inter = scaling(inter_mat)
    diff_matrix = (inter - normal) / normal
    diff_indices = diff_matrix.reshape(-1).argsort().tolist()  # From small to large
    diff_indices.reverse()  # From large to small

    with open('../data/support_materials/cellular_component.txt') as f:
        loc_list = f.read().split()
    with open('../data/generate_materials/protein_ppi.json') as f:
        protein_list = json.load(f)

    loc_mat = load_npz('../data/generate_materials/loc_matrix.npz').toarray()

    # labeled
    # with open('../data/res/' + data + '/loc_change_record(labeled).csv', 'a') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(["Protein", "Score", "Altered localization", "Original localization"])
    #     for indice in diff_indices:
    #         row, col = indice // len(loc_list), indice % len(loc_list)
    #         if loc_mat[row].sum() != 0 and diff_matrix[row][col] != -1.0:
    #             location = loc_map[loc_list[col]]
    #             ori_loc_idx = np.where(loc_mat[row] == 1)[0]
    #             ori_loc = ','.join([loc_map[i] for i in ori_loc_idx])
    #             score = diff_matrix[row][col]
    #             if score > 0 and location not in ori_loc:  # add loc
    #                 protein = protein_list[row]
    #                 writer.writerow([protein, score, location, ori_loc])
    #                 # res_labeled[protein] = [np.float64(score), location, ori_loc]
    #             if score < 0 and location in ori_loc:  # remove loc
    #                 protein = protein_list[row]
    #                 writer.writerow([protein, score, location, ori_loc])
    #                 # res_labeled[protein] = [np.float64(score), location, ori_loc]

    # res_labeled = {}
    # rank = 1
    # for indice in diff_indices:
    #     row, col = indice // len(loc_list), indice % len(loc_list)
    #     if loc_mat[row].sum() != 0 and diff_matrix[row][col] != -1.0:
    #         location = loc_map[loc_list[col]]
    #         ori_loc_idx = np.where(loc_mat[row] == 1)[0]
    #         ori_loc = ','.join([loc_map[i] for i in ori_loc_idx])
    #         score = diff_matrix[row][col]
    #         if score > 0 and location not in ori_loc:  # add loc
    #             protein = protein_list[row]
    #             if protein in res_labeled.keys():
    #                 res_labeled[protein].append([np.float64(score), location, ori_loc, rank])
    #                 rank += 1
    #             else:
    #                 res_labeled[protein] = [[np.float64(score), location, ori_loc, rank], ]
    #                 rank += 1
    #         if score < 0 and location in ori_loc:  # remove loc
    #             protein = protein_list[row]
    #             if protein in res_labeled.keys():
    #                 res_labeled[protein].append([np.float64(score), location, ori_loc, rank])
    #                 rank += 1
    #             else:
    #                 res_labeled[protein] = [[np.float64(score), location, ori_loc, rank], ]
    #                 rank += 1
    # with open('../data/log/res_labeled.json', 'w') as f:
    #     json.dump(res_labeled, f)


    # all data
    res_alldata = {}
    rank = 1
    with open('../data/res/' + data + '/loc_change_record.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Protein", "Score", "Altered localization", "Normal score", "Perturbation score"])
        for indice in tqdm(diff_indices):
            row, col = indice // len(loc_list), indice % len(loc_list)  # get row and col to find protein
            if diff_matrix[row][col] != -1.0:
                location = loc_map[loc_list[col]]
                score = diff_matrix[row][col]
                normal_score = normal[row][col]
                inter_score = inter[row][col]
                if score > 0:  # add loc
                    protein = protein_list[row]
                    writer.writerow([protein, score, location, normal_score, inter_score])
                    if protein in res_alldata.keys():
                        res_alldata[protein].append([np.float64(score), location, rank, np.float64(normal_score), np.float64(inter_score)])
                        rank += 1
                    else:
                        res_alldata[protein] = [[np.float64(score), location, rank, np.float64(normal_score), np.float64(inter_score)], ]
                        rank += 1
                if score < 0:  # remove loc
                    protein = protein_list[row]
                    writer.writerow([protein, score, location, normal_score, inter_score])
                    if protein in res_alldata.keys():
                        res_alldata[protein].append([np.float64(score), location, rank, np.float64(normal_score), np.float64(inter_score)])
                        rank += 1
                    else:
                        res_alldata[protein] = [[np.float64(score), location, rank, np.float64(normal_score), np.float64(inter_score)], ]
                        rank += 1

    with open('../data/res/' + data + '/res_alldata.json', 'w') as f:
        json.dump(res_alldata, f)


if __name__ == "__main__":
    mat_merge()
    for data in ['GSE27182', 'GSE30931', 'GSE74572']:
        loc_normal = np.load('../data/res/' + data + '/normal_logits.npy')
        loc_inter = np.load('../data/res/' + data + '/perturbation_logits.npy')
        misloc_protein_record(loc_normal, loc_inter, data)