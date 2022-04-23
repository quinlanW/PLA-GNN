import csv
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz


def load_data(non_label=False):

    ppi_data = '../data/generate_materials/PPI_normal.npz'
    label_data = '../data/generate_materials/label_list.json'
    label_mask_data = '../data/support_materials/cellular_component.txt'
    loc_data_path = '../data/generate_materials/loc_matrix.npz'

    ppi = load_npz(ppi_data)
    loc_sum = load_npz(loc_data_path).toarray().sum(0).astype(int).tolist()
    with open(label_data) as f:
        label = json.load(f)
    with open(label_mask_data) as f:
        label_mask = f.read().split()

    inter = list(zip(ppi.row, ppi.col))
    interaction = []
    val_map = {}
    for item in inter:
        a_idx = item[0]
        b_idx = item[1]
        a = label[a_idx][0]
        b = label[b_idx][0]
        a_label = label[a_idx][1]
        b_label = label[b_idx][1]
        if len(a_label):
            label_idx = list(map(label_mask.index, a_label))
            loc_flag = loc_sum.index(min([loc_sum[i] for i in label_idx]))
            loc_str = label_mask[loc_flag]
            val_map[a] = loc_str
        else:
            if non_label:
                val_map[a] = 'none'
        if len(b_label):
            label_idx = list(map(label_mask.index, b_label))
            loc_flag = loc_sum.index(min([loc_sum[i] for i in label_idx]))
            loc_str = label_mask[loc_flag]
            val_map[a] = loc_str
        else:
            if non_label:
                val_map[b] = 'none'

        interaction.append((a, b))

    with open('../data/log/edges.csv', 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['source', 'target'])
        for item in interaction:
            csv_writer.writerow([item[0], item[1]])

    with open('../data/log/nodes.csv', 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'label'])
        for key, value in val_map.items():
            csv_writer.writerow([key, value])


if __name__ == '__main__':
    load_data()


